clear all

% load sumulation results of each methods (data of f(x) and the number of
% iteration)
load SGD_data
load SAG_data
load SGD_iter
load SAG_iter
load fs_opt

colors = hsv(13);

% multiple plot on semilog scale
figure
semilogy((0:SGD_iter-1)/546, SGD_data - fs_opt, 'LineWidth',3,'color',colors(9,:),'LineStyle',':'); hold on;
semilogy((0:SAG_iter-1)/546, SAG_data - fs_opt, 'LineWidth',3,'color',colors(6,:),'LineStyle',':'); hold on;
legend('SGD','SAG','FontSize', 15, 'Location', 'best');
axis tight
xlim([0,5])
ylim([1e-4,1e0])
xlabel('#epochs', 'FontSize',16);
ylabel('$f(\mathbf{x}^k) - f^\star$', 'Interpreter', 'latex', 'FontSize',18)
grid on
