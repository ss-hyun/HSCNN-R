%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% Load GT data

load('GT_data_new_v2.mat') 
load('color_chart_mix_list_4.mat')
% Ref_data contains 1: Wavelength, 2: White background, 3: Dark background
% e.g. if you want to select the wavelength information, you should call as
% wavelength = Ref_data(:,1);
% C_data contains raw data of 18 different colours
%% Data normalization
Ref_data=gt_data{2};% 220706 측정 데이터 확용 각 색상 5번씩 측정
wavelength=Ref_data(:,1);
average_white_bg=mean(Ref_data(:,2:6),2);
white_bg=repmat(average_white_bg,[1 size(Ref_data,2)]);
% dark_bg=repmat(Ref_data(:,3),[1 18]);


%% Parameter
input_channel=20;
output_channel=100;
mix_color = false;   % color_loc = [ 1 5 15 25 32 41 48 53 ];
black = true;
black_valid = true;

mesure_per_color = 5; % 주어진 data(GT_data)에서 color 당 측정 횟수
total_colors = 23;    % 측정 color 기준
val_colors = 3;
color_loc = [ 21 22 23 ];
offset_num = 9;
if offset_num ~= 0
    label_tag = ['_offset'];
end

prefix = ['train'; 'valid'];
numbers_of_datas = [200; 50];
base_size_list = [ 500 500 800 800 ];   % 기본 바탕 이미지 크기
seg_size_list = [ 50 25 40 80 ];        % 구역 segmentation 크기
if black
    label_tag = ['_black_val-'];
    total_colors=total_colors+1;
else
    label_tag = ['_val-'];
end
for i=1:1:val_colors
    label_tag = [label_tag int2str(color_loc(i)) '-'];
end
if black_valid
    val_colors = val_colors+1;
    label_tag = [label_tag 'b-'];
    color_loc = [color_loc total_colors];
end
label_tag = [ label_tag(1:size(label_tag,2)-1) '_input+chann+' int2str(input_channel)];


%% Data sysnthesis
% 121 - 1: wavelength, 2-6: white, 7~: color(23개, 5번 측정)
Normalized_colors=(Ref_data)./(white_bg);
% 1-6 remove
Normalized_colors = Normalized_colors(:,7:121);

% Add black color - 매우 작은 random value 할당
if black
    sz=size(Normalized_colors);
    Normalized_colors = [ Normalized_colors rand(sz(1),5)*1.0e-08];
end

if mix_color
    % +30 - mix color, color_pick_list (30개)
    for i=1:1:length(color_pick_list)
        index = total_colors*5 + 5*(i-1);
        c1 = color_pick_list(2*i-1);
        c2 = color_pick_list(2*i);
        for ii=1:1:5
            Normalized_colors(:, index+ii) = (Normalized_colors(:,2+5*c1+ii)+Normalized_colors(:,2+5*c2+ii))/2;
        end
    end
    total_colors=total_colors+length(color_pick_list);
end

wavelength = wavelength(909:1770);
Normalized_colors = Normalized_colors(909:1770,:);

%% Data viewing code (각 스펙트럼 확인)
% for ii=1:1:23
%     figure(35), plot(wavelength, Normalized_colors(:,(ii*5)+2)), title(num2str(ii)),axis([450 700 0 1])
%     pause()
% end
% 숫자는 read_me.ppt에 있는 색상을 의미함.


%% 450 nm (909) - 700 nm (1770)
for p=1:1:size(prefix,1)
    pre = prefix(p,:);
    save_dir = [ pre '_data/' ];
    name_start = 0;  
    numbers_of_data = numbers_of_datas(p,:);

    %% Parameters (여기서 파라미터 변환 가능)
    if strcmp(pre, 'train') == 1
        start_color = 1;
        end_color = total_colors - val_colors;
    else
        start_color = total_colors - val_colors + 1;
        end_color = total_colors;
    end

    if total_colors < end_color
        disp("check end colors")
        return
    end

    img_size = [ 250 250 ]; % DL에 사용할 이미지 크기
    
    for sz=1:1:length(base_size_list)
        base_size = [ base_size_list(sz) base_size_list(sz) ]; % 기본 바탕 이미지 크기
        seg_size = seg_size_list(sz);

        filtered_w_length=imresize(wavelength,[input_channel,1]);
        % filter_transmittance=[1:1:output_channel]*0.4/output_channel+0.8;
        filter_band_width=round(linspace(15,75,output_channel));

    
        %% Data viewing code for resized data (output channel 개수로 변환된 데이터 확인)
%        w_length=imresize(wavelength, [output_channel,1]);
%        N_colors=imresize(Normalized_colors, [output_channel,size(Normalized_colors,2)]);
%         for ii=1:1:total_colors
%             figure(35), plot(w_length, N_colors(:,ii*5+2)), title(num2str(ii)),ylim([0 1])
%             pause()
%         end
        
        %% Base target synthesis
        % 기본 바탕 이미지를 만든 후 랜덤하게 회전해서 사용.
        vertical_num = floor(base_size(1)/seg_size); 
        row_base=repmat(reshape(repmat([1:vertical_num],seg_size,1),1,[])',[1 seg_size]);
        seg_base_img=[];
        for ff=1:1:vertical_num
            seg_base_img=[seg_base_img, row_base+(ff-1)*vertical_num];
        end
%         figure(35), imagesc(seg_base_img),axis image, colormap('bone')

        % 각 영역에 랜덤 색 부여 (테스트)
%         random_color_img=zeros(size(seg_base_img));
%         for ff=1:1:max(max(seg_base_img))
%             random_color_img(find(seg_base_img==ff))=round(rand(1)*17+1); % 각 영역에 랜덤하게 1-18 사이 숫자 부여
%         end
%         
%         figure(36), imagesc(random_color_img,[0 total_colors]),axis image, colormap('bone'), colorbar
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


        %% offset 추가
        %% 색상 위치 change, list에 포함된 번호 color를 맨 뒤로 보내기
        GT_front = [];
        GT_back = [];
%         N_front = [];
%         N_back = [];

        for nn=1:1:total_colors
            start_index = (nn-1)*5 + 1;
            GT_temp = []; %Filtered_colors(:,start_index:1:start_index+4);
%             N_temp = []; %N_colors(:,start_index:1:start_index+4);
            for ii=1:1:mesure_per_color
                %F_temp(:,(ii-1)*5+1) = Filtered_colors(:, start_index+ii-1);
                %N_temp(:,(ii-1)*5+1) = N_colors(:, start_index+ii-1);
                GT_offset = Normalized_colors(:, start_index+ii-1);
%                 N_offset = N_colors(:, start_index+ii-1);
                offset_min = - min(GT_offset); %, min(N_offset));
                offset_max = 1 - max(GT_offset); %, max(N_offset));
                offset_term = (offset_max - offset_min);
                for jj=1:1:offset_num
                    offset = offset_min + offset_term*rand(1);
                    GT_offset(:,jj+1) = GT_offset(:,1) + offset;
%                     N_offset(:,jj+1) = N_offset(:,1) + offset;
                end
                GT_temp = [ GT_temp GT_offset ];
%                 N_temp = [ N_temp N_offset ];
            end
            if find(color_loc==nn)
                GT_back = [ GT_back GT_temp ];
%                 N_back = [ N_back N_temp ];
            else
                GT_front = [ GT_front GT_temp ];
%                 N_front = [ N_front N_temp ];
            end
        end
        GT_colors = [ GT_front GT_back ];
        if (min(min(GT_colors)) < 0) || (min(max(GT_colors)) > 1)
            disp("check offset");
            return
        end
%         N_colors = [ N_front N_back ];

        
        %% Reshape_GT data
        Normalized_colors = GT_colors;
        w_length=imresize(wavelength, [output_channel,1]);
        N_colors=imresize(Normalized_colors, [output_channel,size(Normalized_colors,2)]);
        if (min(min(N_colors)) < 0) || (min(max(N_colors)) > 1)
            disp("check force normalized data");
            return
        end


        %% Filter generation
        filter_pos=round(linspace(1,792,input_channel)); % 909-1770 >> 909-1700
        filters=zeros(length(wavelength),input_channel);
%         length(filter_pos)
        for ff=1:1:length(filter_pos)
            temp_filter=normpdf([1:1:length(wavelength)],filter_pos(ff),filter_band_width(ff));
            temp_filter=temp_filter/max(max(temp_filter));
%             figure(12), plot(wavelength,temp_filter),axis([430 720 0 1])
%             hold on
%             pause(0.1)
            filters(:,ff)=temp_filter';
        end

        %% Filter applied spectrum
        Filtered_colors=[];
        for cc=1:1:size(Normalized_colors,2)
            temp_color=Normalized_colors(:,cc);
%             figure(35), plot(wavelength,temp_color),axis([450 700 0 1])
%             pause(0.1)
            temp_filtered_color=zeros(input_channel,1);
            for tt=1:1:size(filters,2)
                temp_filtered_color(tt,1)=sum(sum(temp_color.*filters(:,tt)))/sum(sum(squeeze(filters(:,tt)>0.45)));
        %         figure(34),plot(wavelength,temp_color.*filters(:,tt)),axis([400 700 0 1])
                %         
                %         pause()
            end
%             
%                 figure(36), plot(filtered_w_length, temp_filtered_color,'o'),ylim([0 1])
%                 hold on
%                 n=floor((cc-2)/5);
%                 t=[num2str(n) ' (' num2str((cc-1)-5*n) ')'];
%                 figure(36), plot(w_length, N_colors(:,cc)), title(t),ylim([0 1])
%                 hold off
%                 pause()
            Filtered_colors(:,cc)=temp_filtered_color;
        end

% for i=1:1:23
%     ii = 2+5*i
%     figure(12),
%     subplot(1,2,1),plot(N_colors(:,ii))
%     subplot(1,2,2),plot(Filtered_colors(:,ii))
%     pause()
% end
%            return
%         N_colors = N_colors(:,122:271);
%         Filtered_colors = Filtered_colors(:,122:271);
%         name = ['result_data/mix+4_data_input+chann+',int2str(input_channel),'.mat'];
%         save(name,'w_length','N_colors', 'filtered_w_length', 'Filtered_colors','-v7.3')
%         return

        
        num_per_color = (offset_num+1)*mesure_per_color;
        %% Data generation
        for nn=1:1:numbers_of_data
            % step1: synthesizing base image with random colors
        
            % 각 영역에 랜덤 색 부여
            random_color_img=zeros(size(seg_base_img));
            for ff=1:1:max(max(seg_base_img))
                random_color_img(find(seg_base_img==ff))=round(rand(1)*(end_color-start_color)+start_color); % 각 영역에 랜덤하게 숫자 부여
            end
            rotated_seg_img=imrotate(random_color_img,rand(1)*360-180);
            c_pos=[round(size(rotated_seg_img,1)/2),round(size(rotated_seg_img,2)/2)];
            cropped_img=rotated_seg_img(c_pos(1)-img_size(1)/2:c_pos(1)+img_size(1)/2-1,c_pos(2)-img_size(2)/2:c_pos(2)+img_size(2)/2-1);
%             figure(36), imagesc(cropped_img),axis image, colormap('bone'), colorbar
            
            % step2: GT data and filtered data synthesis
            GT_data=zeros(img_size(1),img_size(2),output_channel);
            Filtered_data=zeros(img_size(1),img_size(2),input_channel);
            
            for cc=start_color:1:end_color
                % 각 색상이 mesure_per_color(5)번씩 측정 되었고, 한 색상당 offset_num(4)개의 offset이 추가되므로 때문에 0~num_per_color-1(24) 사이의 랜덤 넘버를 생성해 활용
                rand_measure_num=round(rand(1)*(num_per_color-1));
                selected_area = squeeze(cropped_img==cc);
                GT_color_matrix=zeros(img_size(1),img_size(2),output_channel);
                Filtered_color_matrix=zeros(img_size(1),img_size(2),input_channel);
                if find(selected_area>0)
                    % 1-(1+num_per_color-1) 1번 색상, (1+num_per_color)-(1+num_per_color*2-1) 2번 색상
                    % 색상이 num_per_color개 있기 때문에 1+(cc-1)*num_per_color+rand_measure_num index 사용
                    GT_color_matrix=repmat(reshape(N_colors(:,1+(cc-1)*num_per_color+rand_measure_num),[1,1,output_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,output_channel]);
                    Filtered_color_matrix=repmat(reshape(Filtered_colors(:,1+(cc-1)*num_per_color+rand_measure_num),[1,1,input_channel]),[img_size(1),img_size(2),1]).*repmat(selected_area,[1,1,input_channel]);
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
%             figure(37), plot(squeeze(Filtered_data(100,100,:)))
            % 노이즈 추가
%             Noise_data=rand(size(Filtered_data,1),size(Filtered_data,2),size(Filtered_data,3))*0.015;%0.015 이하의 random noise 추가
%             Filtered_data=Filtered_data+Noise_data;
            
%             figure(38),
%             subplot(1,2,1), plot(squeeze(GT_data(100,100,:)),'o')
%             subplot(1,2,2), plot(squeeze(Filtered_data(100,100,:)),'o')
            save(strcat(save_dir,num2str(nn+name_start),'.mat'),'GT_data','Filtered_data','-v7.3')
        end

        %% Data viewing (데이터 확인용)
%         target_area=[120,100];
%         figure(37),
%         subplot(2,2,1), imagesc(GT_data(:,:,1)),axis image,axis on, colorbar
%         subplot(2,2,2), plot(w_length, squeeze(GT_data(target_area(1),target_area(2),:)))
%         subplot(2,2,3), imagesc(Filtered_data(:,:,1)),axis image,axis on, colorbar
%         subplot(2,2,4), plot(filtered_w_length, squeeze(Filtered_data(target_area(1),target_area(2),:)),'o')
%         pause()

        name_start = name_start + numbers_of_data;
    end
end

generate_data_rgb_ntire(label_tag, input_channel, output_channel, numbers_of_datas(1), numbers_of_datas(2))
