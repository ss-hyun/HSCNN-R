%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% Load Blood GT data

load('Blood_data.mat') 
% Ref_data contains 1: Wavelength, 2: BG1, 3: 100% oxy blood, 4: deoxy
% blood, 5: BG2, 6: 100% oxy blood#2, 7 deoxy blood #2
%% Data check
% wavelength=(Blood_data(10:end-1,1));
wavelength=imresize(Blood_data(10:end-1,1),[3205,1]);
Normalized_blood={};
Normalized_blood{1}=imresize(Blood_data(10:end-1,3)./Blood_data(10:end-1,2),[3205 1]);
Normalized_blood{2}=imresize(Blood_data(10:end-1,4)./Blood_data(10:end-1,2),[3205 1]);
Normalized_blood{3}=imresize(Blood_data(10:end-1,6)./Blood_data(10:end-1,5),[3205 1]);
Normalized_blood{4}=imresize(Blood_data(10:end-1,7)./Blood_data(10:end-1,5),[3205 1]);

%% Data viewing code (각 스펙트럼 확인)

% for ii=1:1:4
%     figure(3), 
%     plot(wavelength,(Normalized_blood{ii})),axis([400 700 0 0.7]), hold on % reflectance 학습 데이터로 활용
%     figure(4), 
%     plot(wavelength,abs(-log10(Normalized_blood{ii}))),axis([400 700 0 3]), hold on % absrobance data 확인용
%     pause()
% end
% hold off

%%
Normalized_blood2=cell2mat(Normalized_blood);
% 450 nm (909) - 700 nm (1770)
numbers_of_data=100;
base_size=[500 500]; % 기본 바탕 이미지 크기
img_size=[250 250]; % DL에 사용할 이미지 크기
seg_size=100;
total_colors=4; % 혈액 4종류
input_channel=30; 
output_channel=100;



filtered_w_length=imresize(wavelength(909:1770),[input_channel,1]);
filter_transmittance=[1:1:output_channel]*0.4/output_channel+0.8;
filter_band_width=round(linspace(15,75,output_channel));
%% Reshape_GT data
w_length=imresize(wavelength(909:1770), [output_channel,1]);
N_colors=imresize(Normalized_blood2(909:1770,:), [output_channel,size(Normalized_blood2,2)]);


%% Data viewing code for resized data (output channel 개수로 변환된 데이터 확인)
% for ii=1:1:total_colors
%     figure(35), plot(w_length, N_colors(:,ii)), title(num2str(ii)),ylim([0 1]), hold on
%     pause()
% end
% hold off

%% Base target synthesis
% 기본 바탕 이미지를 만든 후 랜덤하게 회전해서 사용.
vertical_num = floor(base_size(1)/seg_size); 
row_base=repmat(reshape(repmat([1:vertical_num],seg_size,1),1,[])',[1 seg_size]);
seg_base_img=[];
for ff=1:1:vertical_num
    seg_base_img=[seg_base_img, row_base+(ff-1)*vertical_num];
end
% figure(35), imagesc(seg_base_img),axis image, colormap('bone')

%% 각 영역에 랜덤 색 부여 (테스트)
% random_color_img=zeros(size(seg_base_img));
% for ff=1:1:max(max(seg_base_img))
%     random_color_img(find(seg_base_img==ff))=round(rand(1)*17+1); % 각 영역에 랜덤하게 1-18 사이 숫자 부여
% end
% 
% figure(36), imagesc(random_color_img,[0 total_colors]),axis image, colormap('bone')
%% Rotation test (테스트)
% 
% for ii=1:1:10
%     rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
%     c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
%     cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
%     figure(36), subplot(1,2,1), imagesc(rotated_seg_img),axis image,axis off
%     figure(36), subplot(1,2,2), imagesc(cropped_img),axis image,axis off
%     pause(0.1)
% end

%% Filter generation
filter_pos=round(linspace(909,1700,input_channel));
filters=zeros(length(wavelength),input_channel);
for ff=1:1:length(filter_pos)
    temp_filter=normpdf([1:1:length(wavelength)],filter_pos(ff),filter_band_width(ff));
    temp_filter=temp_filter/max(max(temp_filter));
%     figure(12), plot(wavelength,temp_filter),axis([400 700 0 1])
%     hold on
%     pause(0.1)
    filters(:,ff)=temp_filter';
end

%% Filter applied spectrum
Filtered_colors=[];
for cc=1:1:size(Normalized_blood2,2)
    temp_color=Normalized_blood2(:,cc);
%     figure(35), plot(wavelength,temp_color),axis([450 700 0 1])
    temp_filtered_color=zeros(input_channel,1);
    for tt=1:1:size(filters,2)
        temp_filtered_color(tt,1)=sum(sum(temp_color.*filters(:,tt).*squeeze(filters(:,tt)>0.35)))/sum(sum(squeeze(filters(:,tt)>0.35)));
%         figure(34),plot(wavelength,temp_color.*filters(:,tt)),axis([400 700 0 1])
        %         
        %         pause()
    end
    
    %     figure(36), plot(filtered_w_length, temp_filtered_color,'o'),ylim([0 1])
    %     pause()
    Filtered_colors(:,cc)=temp_filtered_color;
end

save(strcat('result_data/','blood_data.mat'),'w_length','N_colors', 'filtered_w_length', 'Filtered_colors','-v7.3')
return
%% Data generation
for nn=1:1:numbers_of_data
    % step1: synthesizing base image with random colors

    % 각 영역에 랜덤 색 부여
    random_color_img=zeros(size(seg_base_img));
    for ff=1:1:max(max(seg_base_img))
        random_color_img(find(seg_base_img==ff))=round(rand(1)*(total_colors-1)+1); % 각 영역에 랜덤하게 1-4 사이 숫자 부여
    end
    rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
    c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
    cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
    figure(36), imagesc(cropped_img,[0 total_colors]),axis image, colormap('bone')

    % step2: GT data and filtered data synthesis
    GT_data=zeros(img_size(1),img_size(2),output_channel);
    Filtered_data=zeros(img_size(1),img_size(2),input_channel);
    
    for cc=1:1:total_colors
        selected_area = squeeze(cropped_img==cc);
        GT_color_matrix=zeros(img_size(1),img_size(2),output_channel);
        Filtered_color_matrix=zeros(img_size(1),img_size(2),input_channel);
        if find(selected_area>0)
            % data 형태가 1열 wavelength, 2-6 wihte 7-11 1번 색상, 12-16 2번
            % 색상----- 이기 때문에 cc*5+1+rand_measure_num index 사용
            GT_color_matrix=repmat(reshape(N_colors(:,cc),[1,1,output_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,output_channel]);
            Filtered_color_matrix=repmat(reshape(Filtered_colors(:,cc),[1,1,input_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,input_channel]);
%             figure(35), subplot(1,2,1),imagesc(selected_area),axis image
%             subplot(1,2,2),imagesc(GT_color_matrix(:,:,1)),axis image
%             pause()

       
        end
        
        GT_data=GT_data+GT_color_matrix;
        Filtered_data=Filtered_data+Filtered_color_matrix;
%         figure(35), subplot(1,2,1),imagesc(selected_area),axis image
%         subplot(1,2,2),imagesc(GT_data(:,:,1),[0 1]),axis image
%         pause()
        
    end
    figure(37), plot(squeeze(Filtered_data(100,100,:)))
    % 노이즈 추가
    Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*0.015;%0.015 이하의 random noise 추가
    Filtered_data=Filtered_data+Noise_data;
    
    figure(38),
    subplot(1,2,1), plot(squeeze(GT_data(100,100,:)),'o')
    subplot(1,2,2), plot(squeeze(Filtered_data(100,100,:)),'o')
%     save(strcat(num2str(nn),'.mat'),'GT_data','Filtered_data','-v7.3')
end
%% Data viewing (데이터 확인용)
% target_area=[120,100];
% figure(37),
% subplot(2,2,1), imagesc(GT_data(:,:,1)),axis image,axis off
% subplot(2,2,2), plot(w_length, squeeze(GT_data(target_area(1),target_area(2),:)))
% subplot(2,2,3), imagesc(Filtered_data(:,:,1)),axis image,axis off
% subplot(2,2,4), plot(filtered_w_length, squeeze(Filtered_data(target_area(1),target_area(2),:)),'o')
