python prepare_data.py ^
-f ./data/trips_bulk_preprocess.csv ^
-e timestamp ^
-test 0.3 ^
-or True ^
-n True ^
-t rel_soc ^
-s sequence ^
-o ./data/trips_fullscale_dataset.pickle ^
-ov ./data/trips_fullscale_validate.pickle ^
-sco ./models/standardscaler.pickle