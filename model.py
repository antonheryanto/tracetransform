import torch
import torch.nn as nn
import torch.nn.functional as F
import ttn as T

class JawiNet(nn.Module):
    def __init__(self, feature_size, is_multi = False, hidden1=512, hidden2=256, length = 7):
        super(JawiNet, self).__init__()
        self.is_multi = is_multi
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        if not self.is_multi:
            #self.subword = nn.Linear(hidden2, 217)
            self.subword = nn.Linear(hidden2, 10)
            self.loss = F.cross_entropy
        else:
            self.letter_length = nn.Linear(hidden2, length)
            self.letters = nn.ModuleList([nn.Linear(hidden2, 51) for i in range(length)])
            self.loss = self.get_loss

    def forward(self, x):
        if self.features != None:
            x = self.features(x)
            # print("feature size", x.shape)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if not self.is_multi:
            return self.subword(x)

        return self.letter_length(x), [ letter(x) for letter in self.letters]

    def feature_init(self, size_in, size_out, stride=2):
        return nn.Sequential(
            nn.Conv2d(size_in, size_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(size_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=stride, padding=1),
            nn.Dropout(0.2)
        )

    def get_loss(self, lengths_predicted, letters_predicted, lengths, letters):
        loss = 0#F.cross_entropy(lengths_predicted, lengths)
        letters = letters.permute(1, 0)
        for i in range(7):
             loss += F.cross_entropy(letters_predicted[i], letters[i])

        return loss
    
class FFNNTF(JawiNet):
    def __init__(self, is_multi = False, hidden1=512, hidden2=256):
        #super(FFNNTF, self).__init__(4 * 64 * 64, is_multi, hidden1, hidden2)
        super(FFNNTF, self).__init__(3 * 256 * 192, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            nn.BatchNorm2d(3)
            #nn.BatchNorm2d(4)
        )

class FFNNOS(JawiNet):
    def __init__(self, is_multi = False, hidden1=512, hidden2=256):
        super(FFNNOS, self).__init__(16 * 64, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            nn.BatchNorm1d(16)
        )


class ConvNet1(JawiNet):
    def __init__(self, is_multi = False, hidden1=512, hidden2=256):
        super(ConvNet1, self).__init__(48 * 33 * 33, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            self.feature_init(1, 48),
        )


class ConvNet3(JawiNet):
    def __init__(self, is_multi = False, hidden1=512, hidden2=256):
        super(ConvNet3, self).__init__(128 * 18 * 18, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            self.feature_init(1, 48), # 64 -> 33
            self.feature_init(48, 64, 1), # 3 -> 34
            self.feature_init(64, 128), # 34 -> 18
        )


class ConvNet8(JawiNet):
    def __init__(self, is_multi = False, hidden1=512, hidden2=256):
        super(ConvNet8, self).__init__(192 * 7 * 7, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            self.feature_init(1, 48),
            self.feature_init(48, 64, 1),
            self.feature_init(64, 128),
            self.feature_init(128, 160, 1), # 19 -> 19
            self.feature_init(160, 192), # 19 -> 10
            self.feature_init(192, 192, 1), # 10 -> 11
            self.feature_init(192, 192), # 11 -> 6
            self.feature_init(192, 192, 1), # 6 -> 7
        )

class TraceLineNet(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
    #def __init__(self, is_multi = False, hidden1=512, hidden2=128):
        # super(TraceNet, self).__init__(4 * 64 * 64, is_multi, hidden1, hidden2)
        super(TraceLineNet, self).__init__(4 * 32 * 32, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            T.TraceLine2d(64),
            nn.BatchNorm2d(4),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

class TraceAngleNet(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
        super(TraceAngleNet, self).__init__(4 * 32 * 32, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            T.TraceAngle2d(64),
            nn.BatchNorm2d(4),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

class TraceLineAngleNet(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
        super(TraceLineAngleNet, self).__init__(4 * 32 * 32, is_multi, hidden1, hidden2)
        self.features = nn.Sequential(
            T.TraceLineAngle2d(64, 64),
            nn.BatchNorm2d(4),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )


class HybridParallel(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
        super(HybridParallel, self).__init__(13504, is_multi, hidden1, hidden2)
        self.feature_global = nn.Sequential(
            T.TraceLine2d(64),
            nn.BatchNorm2d(4),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

        self.feature_local = nn.Sequential(
            self.feature_init(4, 48),
            self.feature_init(48, 64, 1),
            self.feature_init(64, 128),
            self.feature_init(128, 160, 1), # 19 -> 19
            self.feature_init(160, 192), # 19 -> 10
            self.feature_init(192, 192, 1), # 10 -> 11
            self.feature_init(192, 192), # 11 -> 6
            self.feature_init(192, 192, 1), # 6 -> 7
        )


    def forward(self, x):
        l = self.feature_local(x).view(x.size(0), -1)
        g = self.feature_global(x).view(x.size(0), -1)
        x = torch.cat([l,g],1)

        x = self.classifier(x)
        if not self.is_multi:
            return self.subword(x)

        return self.letter_length(x), [ letter(x) for letter in self.letters]

class HybridBefore(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
        super(HybridBefore, self).__init__(4800, is_multi, hidden1, hidden2)
        self.feature_global = nn.Sequential(
            T.TraceAngle2d(64),
            nn.BatchNorm2d(4),
            nn.ReLU(),            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

        self.feature_local = nn.Sequential(
            self.feature_init(4, 48),
            self.feature_init(48, 64, 1),
            self.feature_init(64, 128),
            self.feature_init(128, 160, 1), # 19 -> 19
            self.feature_init(160, 192), # 19 -> 10
            self.feature_init(192, 192, 1), # 10 -> 11
            self.feature_init(192, 192), # 11 -> 6
            self.feature_init(192, 192, 1), # 6 -> 7
        )

        self.features = nn.Sequential(
            self.feature_global,
            self.feature_local
        )

class HybridAfter(JawiNet):
    def __init__(self, is_multi = False, hidden1=1024, hidden2=256):
        super(HybridAfter, self).__init__(192 * 7 * 7, is_multi, hidden1, hidden2)
        
        self.features = nn.Sequential(
            self.feature_init(4, 48),
            self.feature_init(48, 64, 1),
            self.feature_init(64, 128),
            self.feature_init(128, 160, 1), # 19 -> 19
            self.feature_init(160, 192), # 19 -> 10
            self.feature_init(192, 192, 1), # 10 -> 11
            self.feature_init(192, 192), # 11 -> 6
            self.feature_init(192, 192, 1), # 6 -> 7
        )
        

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = self.feature_global(x)
        x = self.classifier(x)
        if not self.is_multi:
            return self.subword(x)

        return self.letter_length(x), [ letter(x) for letter in self.letters]



def test():
    net = TraceLineAngleNet(is_multi=True)
    print(net)
    label = torch.rand(1, 7).long()
    label_length = torch.tensor([6], dtype=torch.long)
    img = torch.rand(1,1,64,64)
    length, letters = net(img)
    print(length, letters)
    loss = net.loss(length, letters, label_length, label)
    print(loss)
