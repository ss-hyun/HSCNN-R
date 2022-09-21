%% Data synthesis code for deep learning written by JYoon 2022.05.25
close all, clear all, clc

%% color chart 합성 대상 리스트 제작 코드
A = int64.empty;
for i=1:1:30
    r1 = randi([1,18],1);
    r2 = randi([1,18],1);
    while r1==r2
        r2 = randi([1,18],1);
    end
    pos = find(A==r1);
    if ~isempty(pos)
        for ii=1:1:length(pos)
            if mod(pos(ii),2)==1
                while A(pos(ii)+1)==r2
                    r2 = randi([1,18],1);
                end
            else
                while A(pos(ii)-1)==r2
                    r2 = randi([1,18],1);
                end
            end
        end
    end
    A(:,i) = [r1;r2];
end
color_pick_list = A;
save("color_chart_mix_list_5.mat","color_pick_list","-v7.3")

return
