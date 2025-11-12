from translate import translate_patois_csv

translate_patois_csv(
    input_path="datasets/jamaican-patois-nli-dataset-translated-premise.csv",
    output_path="datasets/jamaican-patois-nli-dataset-fully-translated.csv",
    column_to_translate="hypothesis")