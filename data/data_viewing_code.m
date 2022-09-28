%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% GT data viewing code
load('GT_data_new_v2.mat')

%% 2 color mix view
second_measurement = gt_data{2};
wavelength = second_measurement(:,1);
white = second_measurement(:,2);
total_colors=18;

% for ii=1:1:total_colors
%     for jj=1:1:total_colors
%         if ii==jj
%             continue
%         end
%         temp_normalised_sig = (second_measurement(:,ii*5+2)+second_measurement(:,jj*5+2))./white./2;
%         figure(35), plot(wavelength,temp_normalised_sig),axis([450 700 0 1]), title([num2str(ii) '-' num2str(jj)])
%         pause()
%     end
% end

%% color view
for ii=1:1:total_colors
    temp_normalised_sig = second_measurement(:,ii*5+2)./white;
    figure(35), plot(wavelength,temp_normalised_sig),axis([450 700 0 1]), title(num2str(ii))
    hold on
    pause()
end