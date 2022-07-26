%% GT data viewing code
load('GT_data_new_v2.mat')
%%
second_measurement = gt_data{2};
wavelength = second_measurement(:,1);
white = second_measurement(:,2);
total_colors=23;
for ii=1:1:total_colors
    temp_normalised_sig = second_measurement(:,ii*5+2)./white;
    figure(35), plot(wavelength,temp_normalised_sig),axis([450 700 0 1]), title(num2str(ii))
    hold on
    pause()
end