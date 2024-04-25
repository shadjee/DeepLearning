clc
clear all
close all
% Load Dataset
tdata=imageDatastore('\\dataset','includesubfolders',true,'LabelSource','foldername')
count=tdata.countEachLabel;
% Load pretrained network
net=vgg16;
layers=[imageInputLayer([250 250 3])
    net(2:end-3)
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer
    ]
%Split data 80:20
[traindata testdata]=splitEachLabel(tdata,0.8,'randomized');
opt=trainingOptions('adam','LearnRateSchedule',"piecewise",'LearnRateDropFactor',0.5,'LearnRateDropPeriod',5,'MaxEpochs',40,'Plot','training-progress')
training=trainNetwork(tdata,layers,opt)

%Batch Testing
allclass=[];
allscore=[];
for i=1:length(testdata.Labels)
    I=readimage(testdata,i);
    [class score] =classify(training,I);
    allclass=[allclass class];
    allscore=[allscore score];
end
result=horzcat(testdata.Labels,allclass');
figure,
plotconfusion(testdata.Labels,allclass')

% Code for CLAHE
function results = myimfcn(im)
shadow_lab = rgb2lab(im);
max_luminosity = 100;
L = shadow_lab(:,:,1)/max_luminosity;

shadow_adapthisteq = shadow_lab;
shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
results= shadow_adapthisteq;
