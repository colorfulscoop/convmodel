from pytorch_lightning.utilities.cli import LightningCLI
from .lightning_model import LightningConversationModel

cli = LightningCLI(LightningConversationModel)
