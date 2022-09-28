%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% Specify condition
input_channel=3;
output_channel=100;
chart_data_num = 0;
bio_data_num = 0;
blood_data_num = 0;
mix_chart_data_num = 30;
chart_data_path = ['result_data/color_data_input+chann+' int2str(input_channel) '.mat'];
blood_data_path = ['result_data/blood_data_input+chann+' int2str(input_channel) '.mat'];
bio_data_path = ['result_data/bio_data_input+chann+' int2str(input_channel) '.mat'];
mix_data_path = ['result_data/mix+4_data_input+chann+' int2str(input_channel) '.mat'];

seg_size = 50;
term = 30;
padd = 40;
h_limit = 5;

%% Data list
data_tag = ["chart", "blood", "bio", "colorMix"];
data_num = [chart_data_num, blood_data_num, bio_data_num, mix_chart_data_num];
data_path = [chart_data_path, blood_data_path, bio_data_path, mix_data_path, ""];
data_index = [7, 1, 1, 1];
data_index_term = [5, 1, 1, 5];

%% Make and Save each data
for d=1:1:length(data_tag)
    tag = data_tag(d);
    num = data_num(d);
    path = data_path(d);
    start_index = data_index(d);
    index_term = data_index_term(d);
    
    if num == 0
        continue
    end

    load(path);

    base_w_size = (fix((num-1)/h_limit)+1)*(seg_size+term) + 2*padd - term;
    base_h_size = h_limit*(seg_size+term) + 2*padd - term;


    F_color_chart = zeros([base_h_size base_w_size input_channel]);
    N_color_chart = zeros([base_h_size base_w_size output_channel]);

    for j=0:1:fix((num-1)/h_limit)
        for i=0:1:h_limit-1
            order = i + h_limit*j + 1;
            if order > num
                break;
            end
            index = start_index+index_term*(order-1);
%             disp(index)
            for chann=1:1:input_channel
                seg_img = zeros([seg_size seg_size]) + Filtered_colors(chann, index);
                start_i = padd+(term+seg_size)*i;
                start_j = padd+(term+seg_size)*j;
                color = [ 
                            zeros([start_i start_j]), zeros([start_i seg_size]), zeros([start_i base_w_size-start_j-seg_size]); ...
                            zeros([seg_size start_j]), seg_img, zeros([seg_size base_w_size-start_j-seg_size]);
                            zeros([base_h_size-start_i-seg_size start_j]), zeros([base_h_size-start_i-seg_size seg_size]), zeros([base_h_size-start_i-seg_size base_w_size-start_j-seg_size])
                        ];
    
                F_color_chart(:,:,chann) = F_color_chart(:,:,chann) + color;
            end
    
            for chann=1:1:output_channel
                seg_img = zeros([seg_size seg_size]) + N_colors(chann, index);
                start_i = padd+(term+seg_size)*i;
                start_j = padd+(term+seg_size)*j;
                color = [ 
                            zeros([start_i start_j]), zeros([start_i seg_size]), zeros([start_i base_w_size-start_j-seg_size]); ...
                            zeros([seg_size start_j]), seg_img, zeros([seg_size base_w_size-start_j-seg_size]);
                            zeros([base_h_size-start_i-seg_size start_j]), zeros([base_h_size-start_i-seg_size seg_size]), zeros([base_h_size-start_i-seg_size base_w_size-start_j-seg_size])
                        ];
    
                N_color_chart(:,:,chann) = N_color_chart(:,:,chann) + color;
            end
        end
    
        if order > num
            break;
        end
    end
    
    % % Show input color chart
    % for i=1:1:input_channel
    %     figure(35), imagesc(F_color_chart(:,:,i)),axis image, colormap('bone'), colorbar
    %     pause(0.2)
    % end
    % 
    % % Show output color chart
    % for i=1:1:output_channel
    %     figure(35), imagesc(N_color_chart(:,:,i)),axis image, colormap('bone'), colorbar
    %     pause(0.1)
    % end
    
    save_dir = 'result_data/';
    input_name = 'input_' + tag + '_chann+' + int2str(input_channel) + '.mat';
    gt_name = 'GT_' + tag + '.mat';
    save(strcat(save_dir,input_name),'F_color_chart', 'filtered_w_length','-v7.3')
    save(strcat(save_dir,gt_name),'N_color_chart', 'w_length','-v7.3')

end
