download-model-data: 
	@echo "Downloading model data"
	@python models/ingredients/data/download.py
validate-model-data:
	@echo "Validating model data"
	@python models/ingredients/validate.py
train-model:
	@echo "Training model"
	@python models/ingredients/train_model.py
eval-model:
	@echo "Eval model"
	@python tests/eval_model.py
test-model-fresh: download-model-data validate-model-data train-model eval-model
	@echo "Starting"
	