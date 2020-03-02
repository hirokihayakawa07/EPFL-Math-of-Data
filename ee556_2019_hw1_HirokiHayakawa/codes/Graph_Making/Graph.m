clear all
% load sumulation results of each methods (data of f(x) and the number of
% iteration)
load GD_data
load AdaGrad_data
load ADAM
load fs_opt

colors = hsv(13);

% multiple plot on semilog scale
figure
loglog(GD_data - fs_opt,'LineWidth',3,'color',colors(1,:)); hold on;
loglog(AdaGrad_data - fs_opt,'LineWidth',3,'color',colors(9,:));
loglog(ADAM_data - fs_opt,'LineWidth',3,'color',colors(12,:));
legend('GD','AdaGrad','ADAM','FontSize', 15, 'Location', 'best');
axis tight
ylim([1e-9,1e0])
xlabel('# iteration', 'FontSize',16);
ylabel('f(x) - f^*', 'FontSize',20);
grid on
hold off