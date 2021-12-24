clear all
close all
clc
file_path =  'D:\study\4\ELEC0134_AMLS\dataset\image\';   % Input images
fileID = fopen('D:\study\4\ELEC0134_AMLS\dataset\label.csv'); % Input name of images and types of tumor
file = textscan(fileID,'%s%s','Delimiter',',','HeaderLines',1); 
F = file{1,1}; % name of image
F(:,2) = file{1,2}; % types of tumor
data = [];
Num_train = 0.85*3000;
Num_test = 0.15*3000;
% Num_train = 3000;   % When using the addition test set
% Num_test = 200;
Num_correct = 0;
% Num_close_point = ceil(Num_train^0.5);
predicts = [];
Accuracy = 0;
if length(F(:,1))  > 0   
    for j = 1:length(F(:,1))  
        F{j,3} = imread(strcat(file_path,F{j,1}));  % Input 512*512 images
        i = string(F(j,2));
        switch i                     % Convert types of tumor to number
            case 'no_tumor'
                F(j,4) = {0};
            case 'meningioma_tumor'
                F(j,4) = {1};
            case 'glioma_tumor'
                F(j,4) = {1};
            case 'pituitary_tumor'
                F(j,4) = {1};
            otherwise
                F(j,4) = {2};
        end
    end  
end
fclose(fileID);


% %% Testing data

%/ When using the addition test set /%

% file_path1 =  'D:\study\4\ELEC0134_AMLS\test\image\';   % Input images
% fileID1 = fopen('D:\study\4\ELEC0134_AMLS\test\label.csv'); % Input name of images and types of tumor
% file1 = textscan(fileID1,'%s%s','Delimiter',',','HeaderLines',1); 
% F1 = file1{1,1}; % name of image
% F1(:,2) = file1{1,2}; % types of tumor
% if length(F1(:,1))  > 0   
%     for j = 1:length(F1(:,1))  
%         F1{j,3} = imread(strcat(file_path1,F1{j,1}));  % Input 512*512 images
%         i = string(F1(j,2));
%         switch i                     % Convert types of tumor to number
%             case 'no_tumor'
%                 F1(j,4) = {0};
%             case 'meningioma_tumor'
%                 F1(j,4) = {1};
%             case 'glioma_tumor'
%                 F1(j,4) = {1};
%             case 'pituitary_tumor'
%                 F1(j,4) = {1};
%             otherwise
%                 F1(j,4) = {2};
%         end
%     end  
% end
% fclose(fileID1);

%% Processing data of graphs
for j = 1:3000 % Downsampling the images
    F{j,5} = Downsampling_average(F{j,3},256); % Average pooling
    M = [];
    m = F{j,5};
    for i = 1:256
        M = [M m(i,:)];  % Transfer data from ceil to matrix
    end
    F{j,6} = M;
    data(j,:) = M';
    label(j,1) = F{j,4}; % labels of images
end

data_train = data(1:Num_train,:); % training data
data_test= data(end-Num_test+1:end,:); % testing data
label_train = label(1:Num_train,:); % labels corresponding to training data
label_test = label(end-Num_test+1:end,:); % true labels corresponding to testing data

% % Testing data

%/ When using the addition test set /%

% for j = 1:200   % Downsampling the images
%     F1{j,5} = Downsampling_average(F1{j,3},256); % Average pooling
%     M = [];
%     m = F1{j,5};
%     for i = 1:256
%         M = [M m(i,:)];  % Transfer data from ceil to matrix
%     end
%     F1{j,6} = M;
%     data1(j,:) = M';
%     label1(j,1) = F1{j,4}; % labels of images
% end
% data_train = data; % training data
% data_test= data1; % testing data
% label_train = label; % labels corresponding to training data
% label_test = label1; % true labels corresponding to testing data

%% Classify the existence of tumor
Num_close_point = 1;
predicts = zeros(2,Num_test);
for i = 1:Num_test
    predicts(i) = kNNClassifier(data_test(i,:),data_train,label_train,Num_close_point); % Prediction of types of tumor
end

Num_correct = 0; 
for j = 1:Num_test % Checking accuarcy
    if label_test(j) == predicts(j)
        Num_correct = Num_correct +1;
    end
end
Accuracy = Num_correct/Num_test;

for a = 1:450
predicts(2,a) = F{end-450+a,4};
end
%% Functions
function y = Downsampling_average(x,a)
y = zeros(a);
for i = 1:a
    for j = 1:a
        y(i,j) = (x(2*i-1,2*j-1,1) + x(2*i,2*j-1,1) + x(2*i-1,2*j,1) + x(2*i,2*j,1))/4;
    end
end
end

function predict = kNNClassifier(x,y,l,N)
[a,~] = size(y);
labels = [];
d = zeros(a,1);
for i = 1:a
    d(i) = distance(x,y(i,:));
end
[~,I] = sort(d);
for i = 1:N
    n = I(i);
    labels(i) = l(n);
end
predict = mode(labels);
end

function d = distance(x,y)
[a,b] = size(x);
d = 0;
for i = 1:a
    for j = 1:b
        d = d + (y(i,j) - x(i,j))^2;
    end
end
end