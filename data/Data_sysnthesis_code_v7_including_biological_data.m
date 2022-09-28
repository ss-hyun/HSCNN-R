%% Data synthesis code for deep learning written by JYoon 2022.09.06
close all, clear all, clc

%% Color chart data synthesis

%% Parameters (여기서 파라미터 변환 가능)

% 450 nm (909) - 700 nm (1770)
% numbers_of_data=100;
base_size=[500 500]; % 기본 바탕 이미지 크기
img_size=[250 250]; % DL에 사용할 이미지 크기
seg_size=25; % 더 작게 조정하면 여러개 컬러 데이터 학습 가능
total_colors=18;
input_channel=50; 
output_channel=100;


mode_setting = 1; %1: Reflection, 2: Absorption, 3: Raw sample signal (normalization 과정 없는 신호)

% % m_path='D:\Projects\015_Deep_learning_based_HSI\Synthetic_code\data_220903' % 데이터 저장경로
% % data_path = 'D:\Projects\015_Deep_learning_based_HSI\Synthetic_code';
% numbers_of_data=500 % Training 데이터 생성 갯수
% numbers_of_test=5; % Test 용 데이터 수
% num_of_color_selected=15; % Training에 사용할 색상 수
% dataset_num = 2; % 데이터 셋 갯수, 설정한 숫자만큼 폴더가 생성됨
% noise_level = 0.05; % 최소 값의 5% 노이즈 추가
% %% Base target synthesis
% % 기본 바탕 이미지를 만든 후 랜덤하게 회전해서 사용.
% vertical_num = floor(base_size(1)/seg_size); 
% row_base=repmat(reshape(repmat([1:vertical_num],seg_size,1),1,[])',[1 seg_size]);
% seg_base_img=[];
% for ff=1:1:vertical_num
%     seg_base_img=[seg_base_img, row_base+(ff-1)*vertical_num];
% end
% figure(35), imagesc(seg_base_img),axis image, colormap('bone')
% %% Load GT data
% % cd(data_path)
% load('GT_data_new_v2.mat') 
% % Ref_data contains 1: Wavelength, 2: White background, 3: Dark background
% % e.g. if you want to select the wavelength information, you should call as
% % wavelength = Ref_data(:,1);
% % C_data contains raw data of 18 different colours
% %% Data preparation
% 
% Ref_data=gt_data{2};% 220706 측정 데이터 확용 각 색상 5번씩 측정
% wavelength=Ref_data(:,1);
% average_white_bg=mean(Ref_data(:,2:6),2);
% white_bg=repmat(average_white_bg,[1 size(Ref_data,2)]);
% 
% switch mode_setting
%     case 1
%         sample_data = (Ref_data)./(white_bg); %반사값 
%     case 2
%         sample_data = abs(-log10((Ref_data)./(white_bg))); %흡수값
%     case 3
%         sample_data = (Ref_data)/max(max(white_bg)); % Raw data를 0-1로 normalization
% end
% 
% % %% Data viewing code (각 스펙트럼 확인)
% % for ii=1:1:23
% %     figure(35), plot(wavelength, sample_data(:,(ii*5)+2)), title(num2str(ii)),axis([400 700 0 1])
% %     pause()
% % end
% % % 숫자는 read_me.ppt에 있는 색상을 의미함.
% 
% 
% %% Reshape_GT data
% w_length=imresize(wavelength(909:1770), [output_channel,1]);
% N_colors=imresize(sample_data(909:1770,:), [output_channel,size(Ref_data,2)]);
% % %% Data viewing code for resized data (output channel 개수로 변환된 데이터 확인)
% % for ii=1:1:total_colors
% %     figure(35), plot(w_length, N_colors(:,ii*5+2)), title(num2str(ii)),ylim([0 1])
% % %     figure(35), plot(w_length, abs(-log10(N_colors(:,ii*5+2)))), title(num2str(ii)),ylim([0 2])
% %     pause(0.1)
% % end
% %% Filter parameters
% 
% filtered_w_length=imresize(wavelength(909:1770),[input_channel,1]);
% filter_transmittance=[1:1:output_channel]*0.4/output_channel+0.8;
% filter_band_width=round(linspace(15,75,output_channel));
% 
% 
% %% Filter generation
% filter_pos=round(linspace(909,1700,input_channel));
% filters=zeros(length(wavelength),input_channel);
% for ff=1:1:length(filter_pos)
%     temp_filter=normpdf([1:1:length(wavelength)],filter_pos(ff),filter_band_width(ff));
%     temp_filter=temp_filter/max(max(temp_filter));
%     figure(12), plot(wavelength,temp_filter),axis([400 700 0 1])
%     hold on
%     pause(0.1)
%     filters(:,ff)=temp_filter';
% end
% %% Filter applied spectrum
% Filtered_colors=[];
% weights=0.45;
% for cc=2:1:size(Ref_data,2)
%     temp_color=sample_data(:,cc);
%     figure(35), plot(wavelength,temp_color),axis([450 700 0 1])
%     temp_filtered_color=zeros(input_channel,1);
%     for tt=1:1:size(filters,2)
%         temp_filtered_color(tt,1)=sum(sum(temp_color.*filters(:,tt)))/sum(sum(squeeze(filters(:,tt)>weights)));
%         
%     end
%     
%     Filtered_colors(:,cc)=temp_filtered_color;
% end
% 
% %% Data generation
% cd(m_path);
% test_color=total_colors-num_of_color_selected;
% 
% for fol=1:1:dataset_num
%     cd(m_path)
%     mkdir(num2str(fol))
%     rand_color_index=randperm(total_colors); % 각 데이터셋마다 랜덤하게 트레이닝에 쓸 color 선택
%     cd(num2str(fol))
%     mkdir('training')
%     mkdir('test')
%     cd('training')
%     
%     % training data set generator
%     for nn=1:1:numbers_of_data+numbers_of_test % training 데이터 생성 후 같은 데이터 셋으로 테스트 데이터 생성 (seen data)
%         
%         % step1: 각 영역에 랜덤 색 부여
%         random_color_img=zeros(size(seg_base_img));
%         for ff=1:1:max(max(seg_base_img))
%             random_color_img(find(seg_base_img==ff))=rand_color_index(round(rand(1)*(num_of_color_selected-1)+1)); % 각 영역에 1-선택된 숫자 사이 랜덤 숫자 생성
%         end
%         rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
%         c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
%         cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
%         
%         % step2: GT data and filtered data synthesis
%         GT_data=zeros(img_size(1),img_size(2),output_channel);
%         Filtered_data=zeros(img_size(1),img_size(2),input_channel);
%         
%         for cc=1:1:total_colors
%             % 각 색상이 5번씩 측정 되었기 때문에 1-5 사이의 랜덤 넘버를 생성해 활용
%             rand_measure_num=round(rand(1)*4+1);
%             selected_area = squeeze(cropped_img==cc);
%             GT_color_matrix=zeros(img_size(1),img_size(2),output_channel);
%             Filtered_color_matrix=zeros(img_size(1),img_size(2),input_channel);
%             if find(selected_area>0)
%                 % data 형태가 1열 wavelength, 2-6 wihte 7-11 1번 색상, 12-16 2번
%                 % 색상----- 이기 때문에 cc*5+1+rand_measure_num index 사용
%                 
%                 GT_color_matrix=repmat(reshape(N_colors(:,cc*5+1+rand_measure_num),[1,1,output_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,output_channel]);
%                 Filtered_color_matrix=repmat(reshape(Filtered_colors(:,cc*5+1+rand_measure_num),[1,1,input_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,input_channel]);
%             end
%             
%             GT_data=GT_data+GT_color_matrix;
%             Filtered_data=Filtered_data+Filtered_color_matrix;
%         end
%         
%         % 노이즈 추가
%         Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*mean(min(min(Filtered_data)))*noise_level;%최소 값의 5% random noise 추가
%         Filtered_data=Filtered_data+Noise_data;
%         if nn<numbers_of_data+1
%             cd(m_path)
%             cd(num2str(fol))
%             cd('training')
%             save(strcat(num2str(nn),'.mat'),'GT_data','Filtered_data','num_of_color_selected','rand_color_index','-v7.3')
%         else
%             cd(m_path)
%             cd(num2str(fol))
%             cd('test')
%             save(strcat('seen',num2str(nn-numbers_of_data),'.mat'),'GT_data','Filtered_data','num_of_color_selected','rand_color_index','-v7.3')
%         end
%         
%     end
%     
%     
%     % test set 학습에 쓰이지 않은 colorchart로 test용 data 생성 (unseen data)
%     cd(m_path)
%     cd(num2str(fol))
%     cd('test')
%     mkdir('output')
%     
%     
%     for nn=1:1:numbers_of_test
%         
%         % step1: synthesizing base image with random colors
%         
%         % 각 영역에 랜덤 색 부여
%         random_color_img=zeros(size(seg_base_img));
%         for ff=1:1:max(max(seg_base_img))
%             %         random_color_img(find(seg_base_img==ff))=round(rand(1)*(total_colors-1)+1); % 각 영역에 랜덤하게 1-23 사이 숫자 부여
%             random_color_img(find(seg_base_img==ff))=rand_color_index(num_of_color_selected+round(rand(1)*(test_color-1)+1)); % 각 영역에 테스트용 랜덤 숫자 생성
%         end
%         rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
%         c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
%         cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
%         
%         % step2: GT data and filtered data synthesis
%         GT_data=zeros(img_size(1),img_size(2),output_channel);
%         Filtered_data=zeros(img_size(1),img_size(2),input_channel);
%         
%         for cc=1:1:total_colors
%             % 각 색상이 5번씩 측정 되었기 때문에 1-5 사이의 랜덤 넘버를 생성해 활용
%             rand_measure_num=round(rand(1)*4+1);
%             selected_area = squeeze(cropped_img==cc);
%             GT_color_matrix=zeros(img_size(1),img_size(2),output_channel);
%             Filtered_color_matrix=zeros(img_size(1),img_size(2),input_channel);
%             if find(selected_area>0)
%                 % data 형태가 1열 wavelength, 2-6 wihte 7-11 1번 색상, 12-16 2번
%                 % 색상----- 이기 때문에 cc*5+1+rand_measure_num index 사용
%                 GT_color_matrix=repmat(reshape(N_colors(:,cc*5+1+rand_measure_num),[1,1,output_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,output_channel]);
%                 Filtered_color_matrix=repmat(reshape(Filtered_colors(:,cc*5+1+rand_measure_num),[1,1,input_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,input_channel]);
%                 %             figure(35), subplot(1,2,1),imagesc(selected_area),axis image
%                 %             subplot(1,2,2),imagesc(GT_color_matrix(:,:,1)),axis image
%                 %             pause()
%                 
%                 
%             end
%             
%             GT_data=GT_data+GT_color_matrix;
%             Filtered_data=Filtered_data+Filtered_color_matrix;
%             %         figure(35), subplot(1,2,1),imagesc(selected_area),axis image
%             %         subplot(1,2,2),imagesc(GT_data(:,:,1),[0 1]),axis image
%             %         pause()
%             
%         end
%         %     figure(37), plot(squeeze(Filtered_data(100,100,:)))
%         % 노이즈 추가
%         %     Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*0.015;%0.015 이하의 random noise 추가
%         Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*mean(min(min(Filtered_data)))*noise_level;%최소 값의 5% random noise 추가
%         
%         Filtered_data=Filtered_data+Noise_data;
%         save(strcat('unseen',num2str(nn),'.mat'),'GT_data','Filtered_data','num_of_color_selected','rand_color_index','-v7.3')
%     end
% end
% 
% 
% %% Data viewing (데이터 확인용)
% % target_area=[30,30];
% % figure(37),
% % subplot(2,2,1), imagesc(GT_data(:,:,1)),axis image,axis off
% % subplot(2,2,2), plot(w_length, squeeze(GT_data(target_area(1),target_area(2),:)))
% % subplot(2,2,3), imagesc(Filtered_data(:,:,1)),axis image,axis off
% % subplot(2,2,4), plot(filtered_w_length, squeeze(Filtered_data(target_area(1),target_area(2),:)),'o')

%% Biological test data preparation
% cd(data_path)
load('Biological_data.mat') 
total_sam_num=8; % 샘플 8종류

% Ref_data contains 1: Wavelength, 2: BG1, 3: 100% oxy blood, 4: deoxy
% blood, 5: BG2, 6: 100% oxy blood#2, 7: deoxy blood#2 8: BG for meat 9:
% Lipid, 10:Lipid#2 11:Muscle, 12:Muscle#2
%%

bg_data = [Biological_data(:,2),Biological_data(:,2),Biological_data(:,5),Biological_data(:,5),Biological_data(:,8),Biological_data(:,8),Biological_data(:,8),Biological_data(:,8)];
bio_data = [Biological_data(:,3),Biological_data(:,4),Biological_data(:,6),Biological_data(:,7),Biological_data(:,9),Biological_data(:,10),Biological_data(:,11),Biological_data(:,12)];

switch mode_setting
    case 1
        sample_data = (bio_data)./(bg_data); %반사값 
    case 2
        sample_data = abs(-log10((bio_data)./(bg_data))); %흡수값
    case 3
        sample_data=bio_data;
        sample_data(:,1:2)=sample_data(:,1:2)/max(max(bg_data(:,1)));
        sample_data(:,3:4)=sample_data(:,1:2)/max(max(bg_data(:,3)));
        sample_data(:,5:8)=sample_data(:,1:2)/max(max(bg_data(:,5)));
        % Raw data를 0-1로 normalization
end

%% 데이터 확인
% for ii=1:1:size(sample_data,2)
%     
%     figure(4),
%     plot(Biological_data(:,1),sample_data(:,ii)),axis([400 700 0 1]), hold on
%     pause(0.1)
% end
% hold off

%% 데이터 차원 맞추기
resized_sample_data = zeros(3205,size(sample_data,2));
wavelength=imresize(Biological_data(10:end-1,1),[3205,1]);
for ii=1:1:size(sample_data,2)
    resized_sample_data(:,ii) = imresize(sample_data(10:end-1,ii),[3205 ,1]);
end

bio_sample_data = resized_sample_data(:, 5:8);
save(strcat('./','biological_format_data.mat'),'bio_sample_data', '-v7.3')

blood_sample_data = resized_sample_data(:, 1:4);
save(strcat('./','blood_sample_data.mat'),'blood_sample_data', '-v7.3')
return
%% 데이터 확인
% for ii=1:1:size(sample_data,2)
%     
%     figure(5),
%     plot(wavelength,resized_sample_data(:,ii)),axis([400 700 0 1]), hold on
%     pause(0.1)
% end
% hold off

%%  Reshape_GT data
input_channel=10; 
output_channel=100;

for n=1:1:2
    if n==1
        resized_sample_data = blood_sample_data;
        name = ['result_data/blood_data_input+chann+',int2str(input_channel),'.mat'];
    else
        resized_sample_data = bio_sample_data;
        name = ['result_data/bio_data_input+chann+',int2str(input_channel),'.mat'];
    end


    w_length=imresize(wavelength(909:1770), [output_channel,1]);
    N_colors=imresize(resized_sample_data(909:1770,:), [output_channel,size(resized_sample_data,2)]);
    
    % Filter generation
    filter_pos=round(linspace(909,1700,input_channel));
    filters=zeros(length(wavelength),input_channel);
    % filter_transmittance=[1:1:output_channel]*0.4/output_channel+0.8;
    filter_band_width=round(linspace(15,75,output_channel));
    for ff=1:1:length(filter_pos)
        temp_filter=normpdf([1:1:length(wavelength)],filter_pos(ff),filter_band_width(ff));
        temp_filter=temp_filter/max(max(temp_filter));
    %     figure(12), plot(wavelength,temp_filter),axis([400 700 0 1])
    %     hold on
    %     pause(0.1)
        filters(:,ff)=temp_filter';
    end
    
    % Filter applied spectrum
    Filtered_colors=[];
    filtered_w_length=imresize(wavelength(909:1770),[input_channel,1]);
    for cc=1:1:size(resized_sample_data,2)
        temp_color=resized_sample_data(:,cc);
        %     figure(35), plot(wavelength,temp_color),axis([450 700 0 1])
        temp_filtered_color=zeros(input_channel,1);
        for tt=1:1:size(filters,2)
            temp_filtered_color(tt,1)=sum(sum(temp_color))/sum(sum(squeeze(filters(:,tt)>0.45)));
            
        end
        
    %     figure(36), plot(filtered_w_length, temp_filtered_color,'o'),ylim([0 1]), hold on
    %     
    %     pause(0.1)
        Filtered_colors(:,cc)=temp_filtered_color;
    end
    
    save(name,'w_length','N_colors', 'filtered_w_length', 'Filtered_colors','-v7.3')

end

return
%% Data generation
for fol=1:1:dataset_num
    cd(m_path)
    cd(num2str(fol))
    cd('test')
    for nn=1:1:5
        % step1: synthesizing base image with random colors
        
        % 각 영역에 랜덤 색 부여
        random_color_img=zeros(size(seg_base_img));
        for ff=1:1:max(max(seg_base_img))
            random_color_img(find(seg_base_img==ff))=round(rand(1)*(total_sam_num-1)+1); % 각 영역에 랜덤하게 1-8 사이 숫자 부여
        end
        rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
        c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
        cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
        figure(36), imagesc(cropped_img,[0 total_sam_num]),axis image, colormap('bone')
        
        % step2: GT data and filtered data synthesis
        GT_data=zeros(img_size(1),img_size(2),output_channel);
        Filtered_data=zeros(img_size(1),img_size(2),input_channel);
        
        for cc=1:1:total_sam_num
            selected_area = squeeze(cropped_img==cc);
            GT_color_matrix=zeros(img_size(1),img_size(2),output_channel);
            Filtered_color_matrix=zeros(img_size(1),img_size(2),input_channel);
            if find(selected_area>0)
  
                GT_color_matrix=repmat(reshape(N_colors(:,cc),[1,1,output_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,output_channel]);
                Filtered_color_matrix=repmat(reshape(Filtered_colors(:,cc),[1,1,input_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,input_channel]);
            end
            
            GT_data=GT_data+GT_color_matrix;
            Filtered_data=Filtered_data+Filtered_color_matrix;
            %         figure(35), subplot(1,2,1),imagesc(selected_area),axis image
            %         subplot(1,2,2),imagesc(GT_data(:,:,1),[0 1]),axis image
            %         pause()
            
        end
%         figure(37), plot(squeeze(Filtered_data(100,100,:)))
        % 노이즈 추가
        Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*mean(min(min(Filtered_data)))*noise_level;%최소 값의 5% 노이즈 레벨
        Filtered_data=Filtered_data+Noise_data;
        
        figure(38),
        subplot(1,2,1), plot(squeeze(GT_data(100,100,:)),'o')
        subplot(1,2,2), plot(squeeze(Filtered_data(100,100,:)),'o')
        save(strcat('Biological_data_',num2str(nn),'.mat'),'GT_data','Filtered_data','-v7.3')
    end
end