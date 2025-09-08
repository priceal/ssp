# Save model
torch.save(model.state_dict(), "CNNonelength_20250908A_model.pth")

# Create new model and load states
#newmodel = Multiclass()
#newmodel.load_state_dict(torch.load("iris-model.pth"))
