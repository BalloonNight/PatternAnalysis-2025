from train import Trainer

# Set up trainer
trainer = Trainer()

# Load desired model into trainer
model_name = "Final_Model_0_7756310105323792_1761864592.pt"
trainer.load_model(f"./output/model/{model_name}")

# Run the test set on the trainer
trainer.test(visualise=True)
