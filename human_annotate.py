# review_app_fixed.py
import io, os
import pandas as pd
import streamlit as st

# ---------------- GLOBAL UI STYLING ----------------

st.set_page_config(page_title="Patois Translation Reviewer", layout="wide")

REQUIRED_SRC_COLS = ["premise", "hypothesis"]
ENG_HYPOTHESIS_FALLBACKS = ["english_hypothsis", "english_hypothesis"]

def load_dataframe(path_or_file):
    if isinstance(path_or_file, str):
        df = pd.read_csv(path_or_file)
        name = os.path.basename(path_or_file)
    else:
        df = pd.read_csv(path_or_file)
        name = getattr(path_or_file, "name", "uploaded.csv")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df, name


def ensure_columns(df):
    # existing english-hypothesis detection...
    eng_hyp_col = next((c for c in ENG_HYPOTHESIS_FALLBACKS if c in df.columns), None)
    if eng_hyp_col is None:
        st.error(f"Missing English hypothesis column; expected one of {ENG_HYPOTHESIS_FALLBACKS}")
        st.stop()

    for c in REQUIRED_SRC_COLS + ["english_premise", eng_hyp_col]:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            st.stop()

    # create columns if missing
    for c in [
        "premise_reviewed", "premise_acceptable", "premise_edited",
        "hypothesis_reviewed", "hypothesis_acceptable", "hypothesis_edited",
    ]:
        if c not in df.columns:
            df[c] = False if c.endswith(("_reviewed", "_acceptable")) else ""

    # 🔑 normalize types & NaNs for already-existing sheets
    bool_cols = [
        "premise_reviewed", "premise_acceptable",
        "hypothesis_reviewed", "hypothesis_acceptable",
    ]
    for c in bool_cols:
        df[c] = df[c].fillna(False).astype(bool)

    text_cols = ["premise_edited", "hypothesis_edited"]
    for c in text_cols:
        df[c] = df[c].astype(str)
        df[c] = df[c].replace("nan", "").fillna("")

    return eng_hyp_col


def build_final_columns(df: pd.DataFrame, eng_hyp_col: str) -> pd.DataFrame:
    """
    Create non-destructive 'final' columns that prefer human edits but
    DO NOT modify the original english_* columns.
    """
    # Premise
    prem_edit = df["premise_edited"].astype(str).str.strip()
    df["final_english_premise"] = prem_edit.where(
        prem_edit != "", df["english_premise"]
    )

    # Hypothesis
    hyp_edit = df["hypothesis_edited"].astype(str).str.strip()
    df["final_english_hypothesis"] = hyp_edit.where(
        hyp_edit != "", df[eng_hyp_col]
    )

    return df

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("Reviewer Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
path_input = st.sidebar.text_input("…or load from path")
save_path = st.sidebar.text_input("Save to", value="reviewed_output.csv")
only_unreviewed = st.sidebar.checkbox("Only show unreviewed", value=False)

# init session state
if "df" not in st.session_state:
    if uploaded is not None:
        st.session_state.df, _ = load_dataframe(uploaded)
    elif path_input.strip():
        st.session_state.df, _ = load_dataframe(path_input.strip())
    else:
        st.info("Upload a CSV or provide a path to start.")
        st.stop()
    st.session_state.eng_hyp_col = ensure_columns(st.session_state.df)
    st.session_state.idx = 0
    st.session_state.only_unreviewed = only_unreviewed

# keep current filter choice in state
st.session_state.only_unreviewed = only_unreviewed

df = st.session_state.df
eng_hyp_col = st.session_state.eng_hyp_col
total = len(df)

# index queue (all rows or only unreviewed)
if st.session_state.only_unreviewed:
    indices = df.index[(~df["premise_reviewed"]) | (~df["hypothesis_reviewed"])].tolist()
    if not indices:
        st.success("All rows reviewed 🎉")
        if st.sidebar.button("Save CSV now"):
            df.to_csv(save_path, index=False)
            st.sidebar.success(f"Saved to {save_path}")
        st.stop()
else:
    indices = df.index.tolist()

# clamp idx
st.session_state.idx = max(0, min(st.session_state.idx, len(indices) - 1))
row_idx = indices[st.session_state.idx]
row = df.loc[row_idx]

# ---------------------- NAV BAR ----------------------
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"**Row:** {row_idx+1} / {total}")
with c2: pass
with c3: st.markdown(f"**Queue:** {st.session_state.idx+1} / {len(indices)}")

nav1, nav2, nav3, nav4 = st.columns([1,1,2,2])
with nav1:
    if st.button("⟵ Prev", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
with nav2:
    if st.button("Next ⟶", use_container_width=True):
        st.session_state.idx = min(len(indices)-1, st.session_state.idx + 1)
        st.rerun()
with nav3:
    jump = st.number_input("Jump to row #", min_value=1, max_value=total, value=row_idx+1, step=1)
with nav4:
    if st.button("Go", use_container_width=True):
        target = jump - 1
        if st.session_state.only_unreviewed:
            try:
                st.session_state.idx = indices.index(target)
            except ValueError:
                st.session_state.idx = 0
        else:
            st.session_state.idx = max(0, min(target, len(indices)-1))
        st.rerun()

st.divider()

# ---------------------- REVIEW UI ----------------------
colA, colB = st.columns(2)

# current values
prem_rev = bool(row["premise_reviewed"])
prem_acc = bool(row["premise_acceptable"])
prem_edit = str(row["premise_edited"])

hyp_rev = bool(row["hypothesis_reviewed"])
hyp_acc = bool(row["hypothesis_acceptable"])
hyp_edit = str(row["hypothesis_edited"])

with colA:
    st.subheader("Premise")
    st.text_area("premise (source)", value=str(row["premise"]), height=110, disabled=False)
    st.text_area("english_premise (current)", value=str(row["english_premise"]), height=110, disabled=False)
    prem_rev = st.checkbox("Reviewed (premise)", value=prem_rev, key=f"prem_rev_{row_idx}")
    prem_acc = st.checkbox("Acceptable (premise)", value=prem_acc, key=f"prem_acc_{row_idx}")
    prem_edit = st.text_area("Edited English (premise)", value=prem_edit, height=80, key=f"prem_edit_{row_idx}")

with colB:
    st.subheader("Hypothesis")
    st.text_area("hypothesis (source)", value=str(row["hypothesis"]), height=110, disabled=False)
    st.text_area(f"{eng_hyp_col} (current)", value=str(row[eng_hyp_col]), height=110, disabled=False)
    hyp_rev = st.checkbox("Reviewed (hypothesis)", value=hyp_rev, key=f"hyp_rev_{row_idx}")
    hyp_acc = st.checkbox("Acceptable (hypothesis)", value=hyp_acc, key=f"hyp_acc_{row_idx}")
    hyp_edit = st.text_area("Edited English (hypothesis)", value=hyp_edit, height=80, key=f"hyp_edit_{row_idx}")

# ----- STICKY “MARK BOTH …” TOGGLES (stay highlighted) -----
t1, t2, t3 = st.columns(3)
both_reviewed = t1.toggle("Mark BOTH reviewed", value=False, key=f"both_rev_{row_idx}")
both_acceptable = t2.toggle("Mark BOTH acceptable", value=False, key=f"both_acc_{row_idx}")
# copy_to_edits = t3.toggle("Copy EN → Edited (both)", value=False, key=f"copy_both_{row_idx}")

if both_reviewed:
    prem_rev = hyp_rev = True
if both_acceptable:
    prem_acc = hyp_acc = True


# apply edits to the in-memory df (session state)
df.loc[row_idx, "premise_reviewed"] = bool(prem_rev)
df.loc[row_idx, "premise_acceptable"] = bool(prem_acc)
df.loc[row_idx, "premise_edited"] = str(prem_edit).strip()

df.loc[row_idx, "hypothesis_reviewed"] = bool(hyp_rev)
df.loc[row_idx, "hypothesis_acceptable"] = bool(hyp_acc)
df.loc[row_idx, "hypothesis_edited"] = str(hyp_edit).strip()

st.divider()

def apply_human_edits(df, eng_hyp_col: str) -> pd.DataFrame:
    """
    For every row:
      - if premise_edited is non-empty, copy it into english_premise
      - if hypothesis_edited is non-empty, copy it into the english hypothesis column
    """
    # Premise
    prem_edit = df["premise_edited"].astype(str).str.strip()
    mask_prem = prem_edit != ""
    df.loc[mask_prem, "english_premise"] = prem_edit[mask_prem]

    # Hypothesis
    hyp_edit = df["hypothesis_edited"].astype(str).str.strip()
    mask_hyp = hyp_edit != ""
    df.loc[mask_hyp, eng_hyp_col] = hyp_edit[mask_hyp]

    return df

# ---------------------- SAVE BAR ----------------------
b1, b2, b3 = st.columns([2,2,3])

with b1:
    if st.button("💾 Save (entire file) & Stay", use_container_width=True):
        # build final_* columns on a copy so we don't touch originals in memory
        df_to_save = build_final_columns(st.session_state.df.copy(), eng_hyp_col)
        df_to_save.to_csv(save_path, index=False)
        st.success(f"Saved all rows to {save_path}")

with b2:
    if st.button("💾 Save (entire file) & Next →", use_container_width=True):
        df_to_save = build_final_columns(st.session_state.df.copy(), eng_hyp_col)
        df_to_save.to_csv(save_path, index=False)
        st.session_state.idx = min(len(indices)-1, st.session_state.idx + 1)
        st.rerun()

with b3:
    buf = io.StringIO()
    df_for_download = build_final_columns(st.session_state.df.copy(), eng_hyp_col)
    df_for_download.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download current CSV",
        buf.getvalue(),
        file_name="reviewed_output.csv",
        mime="text/csv",
        use_container_width=True,
    )

