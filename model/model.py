import torch
import torch.nn as nn
import torch.nn.functional as F

class PMG_model(nn.Module):
    ### progressive multi-granularity model
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
        fc3,
        drop3,
        num_coarse,
        num_fine,
        feature_dim,
    ):
        super(PMG_model, self).__init__()

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

        # coarse classification head
        self.coarse_head = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, num_coarse),
        )

        # fine classification head
        self.fine_head = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, num_fine),
        )

        # projection head; used for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Linear(fc1, feature_dim),
        )

        # self.corr_matrix = nn.Parameter(torch.randn(num_fine, num_coarse) * 0.01)
        C_init = self.initialize_C(num_fine, num_coarse)
        self.corr_matrix = nn.Parameter(C_init, requires_grad=True) 

        # fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(num_fine + num_coarse, fc3),
            nn.ReLU(),
            nn.Linear(fc3, num_fine),
        )

    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size

    def initialize_C(self, num_fine, num_coarse):

        C_matrix = torch.zeros(num_fine, num_coarse)
        for i in range(num_fine):
            j = i % num_coarse
            C_matrix[i, j] = 1.0
        noise = torch.randn_like(C_matrix) * 0.02
        return C_matrix + noise


    def forward(self, x, y_fine=None):

        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        ### projection outputs
        outputs_projection = self.projection_head(features)

        ### classification outputs
        outputs_coarse = self.coarse_head(features)
        outputs_fine = self.fine_head(features)

        C_matrix = F.softmax(self.corr_matrix, dim=1)  # soft correlation matrix
        if y_fine is not None:  # y_fine is provided during training. none during inference
            
            y_fine_oh = F.one_hot(y_fine, num_classes=outputs_fine.size(1)).float()
            y_coarse = torch.matmul(y_fine_oh, C_matrix)
        
        else:
            y_coarse = None

        # p_coarse = torch.softmax(outputs_coarse, dim=1)
        # coarse_vote = torch.matmul(p_coarse, C_matrix)

        fusion_input = torch.cat((outputs_fine, outputs_coarse), dim=1)

        outputs_fusion = self.fusion_classifier(fusion_input)

        return {
            'outputs_coarse': outputs_coarse, 
            'outputs_fine': outputs_fine,
            'outputs_fusion': outputs_fusion,
            'outputs_projection': outputs_projection,
            'corr_matrix': C_matrix,
            'y_coarse': y_coarse
        }
