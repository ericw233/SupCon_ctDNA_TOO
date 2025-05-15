import torch
import torch.nn as nn

class SupConModel(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        out1,
        out2,
        conv1,
        pool1,
        drop1,
        conv2,
        pool2,
        drop2,
        fc1,
        fc2,
        drop3,
        feature_dim,
        num_classes,
    ):
        super(SupConModel, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out1),
            nn.Dropout(drop1),
            nn.MaxPool1d(kernel_size=pool1, stride=2),
            nn.Conv1d(
                in_channels=out1,
                out_channels=out2,
                kernel_size=conv2,
                stride=2,
                bias=None,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2),
        )

        self.fc_input_size = self._get_fc_input_size(input_size)

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Linear(fc1, feature_dim),
        )

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, num_classes),
        )

    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # projection classifier output
        projection_output = self.projection_head(features)
        # task classifier output
        task_output = self.task_classifier(features)

        return projection_output, task_output
