from pytorch_lightning.utilities.cli import LightningCLI
from .pl_model import LightningConversationModel

cli = LightningCLI(LightningConversationModel)
