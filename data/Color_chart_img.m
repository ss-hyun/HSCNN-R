%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% GT data
load('result_data/color_data.mat')
c_w_length = w_length;
c_filtered_w_length = filtered_w_length;
c_Filtered_colors = Filtered_colors;
c_N_colors = N_colors;
load('result_data/blood_data.mat')
b_w_length = w_length;
b_filtered_w_length = filtered_w_length;
b_Filtered_colors = Filtered_colors;
b_N_colors = N_colors;
%%

input_channel=30;
output_channel=100;
base_w_size = 450;
base_h_size = 530;
seg_size = 50;
padd = 40;
term = 30;
total_colors = 23;
total_blood = 4;
total = total_blood + total_colors;
start_index = 7;

F_color_chart = zeros([base_w_size base_h_size input_channel]);
N_color_chart = zeros([base_w_size base_h_size output_channel]);

for j=0:1:5
    for i=0:1:4
        order = i + 5*j + 1;
        if order > total
            break;
        elseif order > total_colors
            w_length = b_w_length;
            filtered_w_length = b_filtered_w_length;
            Filtered_colors = b_Filtered_colors;
            N_colors = b_N_colors;
            data_index = order - total_colors;
        else
            w_length = c_w_length;
            filtered_w_length = c_filtered_w_length;
            Filtered_colors = c_Filtered_colors;
            N_colors = c_N_colors;
            data_index = start_index+5*(order-1);
        end
            for chann=1:1:input_channel
                seg_img = zeros([seg_size seg_size]) + Filtered_colors(chann, data_index);
                start_i = padd+(term+seg_size)*i;
                start_j = padd+(term+seg_size)*j;
                color = [ 
                            zeros([start_i start_j]), zeros([start_i seg_size]), zeros([start_i base_h_size-start_j-seg_size]); ...
                            zeros([seg_size start_j]), seg_img, zeros([seg_size base_h_size-start_j-seg_size]);
                            zeros([base_w_size-start_i-seg_size start_j]), zeros([base_w_size-start_i-seg_size seg_size]), zeros([base_w_size-start_i-seg_size base_h_size-start_j-seg_size])
                        ];
    
                F_color_chart(:,:,chann) = F_color_chart(:,:,chann) + color;
            end
    
            for chann=1:1:output_channel
                seg_img = zeros([seg_size seg_size]) + N_colors(chann, data_index);
                start_i = padd+(term+seg_size)*i;
                start_j = padd+(term+seg_size)*j;
                color = [ 
                            zeros([start_i start_j]), zeros([start_i seg_size]), zeros([start_i base_h_size-start_j-seg_size]); ...
                            zeros([seg_size start_j]), seg_img, zeros([seg_size base_h_size-start_j-seg_size]);
                            zeros([base_w_size-start_i-seg_size start_j]), zeros([base_w_size-start_i-seg_size seg_size]), zeros([base_w_size-start_i-seg_size base_h_size-start_j-seg_size])
                        ];
    
                N_color_chart(:,:,chann) = N_color_chart(:,:,chann) + color;
            end
    end

    if order > total
        break;
    end
end

% % Show input color chart
% for i=1:1:input_channel
%     figure(35), imagesc(F_color_chart(:,:,i)),axis image, colormap('bone'), colorbar
%     pause(0.2)
% end

% % Show output color chart
% for i=1:1:output_channel
%     figure(35), imagesc(N_color_chart(:,:,i)),axis image, colormap('bone'), colorbar
%     pause(0.1)
% end

save_dir = 'result_data/';
input_name = 'input_chart+blood.mat';
gt_name = 'GT_chart+blood.mat';
save(strcat(save_dir,input_name),'F_color_chart', 'filtered_w_length','-v7.3')
save(strcat(save_dir,gt_name),'N_color_chart', 'w_length','-v7.3')

return
