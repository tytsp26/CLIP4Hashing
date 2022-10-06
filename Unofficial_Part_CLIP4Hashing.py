'''
Unofficial Implementation of ICMR 2022 Paper 

CLIP4Hashing: Unsupervised Deep Hashing for Cross-Modal Video-Text Retrieval

I tried to implement part of it. Only for academic purpose. All rights are reserved by the original authors.
'''

parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--bit_size', default=512, type=int)
parser.add_argument('--gamma', default = 0.01, type = float)


'''
The Hash Net
'''
class HashNet(nn.Module):
    def __init__(self, bit_size):
        super(HashNet, self).__init__()
        self.hash = nn.Sequential(nn.Linear(512,4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5)
        )
        self.fc_encode = nn.Linear(4096, bit_size)
        
    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            temp_U = U.t()
            max_value, max_index = torch.max(temp_U, dim=1)
            min_value, min_index = torch.min(temp_U, dim=1)
            maxmin_values = torch.stack([max_value, min_value], dim=1).unsqueeze(2)
            temp_U=temp_U.unsqueeze(2)
            dist = torch.cdist(temp_U, maxmin_values, p=1).to('cuda')
            differences=dist[:,:,0]-dist[:,:,1]
            B = torch.zeros(differences.shape).to('cuda')
            B[differences < 0] = 1
            B[differences >= 0] = -1
            B = B.t().to('cuda')
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            U, B = ctx.saved_tensors
            add_g = (U - B) / (B.numel())
            grad = g + args.gamma * add_g
            return grad

    def forward(self, x):
        feat = self.hash(x)
        hid = self.fc_encode(feat)
        code = HashNet.Hash.apply(hid)
        return hid, code

'''
Dynamic Weighting
'''
def dynamic_weighting(S_):
    S_mean = torch.mean(S_)
    S_min = torch.min(S_)
    S_max = torch.max(S_)
    small_threshold = S_mean 
    large_threshold = S_mean

    S_[S_ < small_threshold]= torch.exp( -0.5*(S_[S_ < small_threshold] - small_threshold)/( S_min - small_threshold ) ) * S_[S_ < small_threshold]
    S_[S_ > large_threshold]= torch.exp( 0.5*(S_[S_ > large_threshold] - large_threshold)/( S_max - large_threshold ) ) * S_[S_ > large_threshold]

    S_[S_>1.0]=1.0
    S_[S_<-1.0]=-1.0
    return S_

'''
Training
'''
for video_feature, sentence_feature in data_loader:
    '''
    For the CLIP feature extraction process

    The frame features are extracted by the pre-trained CLIP model (ViT-B/32) at a fixed frame sampling rate like 1 fps. 
    By this way, one video can be represented as one feature vector by averaging. 

    Similar to video features, the sentences' features are extracted by the pre-trained CLIP model (ViT-B/32).
    '''
    optimizer.zero_grad()

    F_I = video_feature.to('cuda')
    F_T = sentence_feature.to('cuda')

    hid_I, code_I = model(F_I)
    hid_T, code_T = model(F_T)

    # Construct cross-modal affinity matrix
    F_I = F.normalize(F_I)
    F_T = F.normalize(F_T)

    S_IT = F_I.mm(F_T.t())
    S_TI = F_T.mm(F_I.t())

    H_I = F.normalize(hid_I)
    H_T = F.normalize(hid_T)

    HI_HI = H_I.mm(H_I.t())
    HT_HT = H_T.mm(H_T.t())
    HI_HT = H_I.mm(H_T.t())
    HT_HI = H_T.mm(H_I.t())

    # Set diagonal elements to 1
    complete_S_IT_diagonal = torch.diag_embed(1 - S_IT.diagonal())
    complete_S_TI_diagonal = torch.diag_embed(1 - S_TI.diagonal())
    S_IT = S_IT + complete_S_IT_diagonal
    S_TI = S_TI + complete_S_TI_diagonal

    S_tilde = 0.5 * S_TI + 0.5 * S_IT
    S = dynamic_weighting(S_tilde)

    intra_loss = F.mse_loss(HI_HI, S) + F.mse_loss(HT_HT, S)
    inter_loss = F.mse_loss(HI_HT, S) + F.mse_loss(HT_HI, S)
    consistency_loss = F.mse_loss(H_I, H_T)

    loss = 0.1 * intra_loss + 1 * inter_loss + 2 * consistency_loss
    loss.backward()

    optimizer.step()