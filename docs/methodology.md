# Methodology

The primary method is intentionally classical and interpretable. It avoids using
policy identity, checkpoint names, XML identifiers, rewards, commands, contacts,
or simulator state during inference. Those fields are used only for labels,
validation, grouping, and analysis.

The leakage-safe model input is the RGB clip plus metadata such as FPS and
camera ID. Grouped splitting prevents clips from the same checkpoint/seed family
from appearing in both train and test.
