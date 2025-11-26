soil_params = xlsread('designed_parameters_for_P.xlsx', 'soil_parameters');
rubber_params = xlsread('designed_parameters_for_P.xlsx', 'designed_rubber_parameters');
ffs = xlsread('designed_parameters_for_P.xlsx', 'designed_filling_fractions');
as = xlsread('designed_parameters_for_P.xlsx', 'designed_periodic_constants');
Es = soil_params(:, 1)*1e6;
Ps = soil_params(:, 2);
rhos = soil_params(:, 3)*1e3;
Er = rubber_params(:, 1)*1e6;
Pr = rubber_params(:, 2);
rhor = rubber_params(:, 3)*1e3;

for i=1:20
tic
model = mphload('P.mph');
filename = ['dispersion_curves_for_P/', num2str(i), '.csv'];

model.param.set('L1', num2str(ffs(i, 1)));
model.param.set('L2', num2str(ffs(i, 2)));
model.param.set('L3', num2str(ffs(i, 3)));
model.param.set('L4', num2str(ffs(i, 4)));
model.param.set('a', num2str(as(i)));
model.param.set('Es', num2str(Es(i)));
model.param.set('Ps', num2str(Ps(i)));
model.param.set('rhos', num2str(rhos(i)));
model.param.set('Er', num2str(Er(i)));
model.param.set('Pr', num2str(Pr(i)));
model.param.set('rhor', num2str(rhor(i)));


model.study('std1').run;

model.result.export.create('plot1', 'pg3', 'glob1', 'Plot');
model.result('pg3').set('window', 'graphics');
model.result('pg3').run;
model.result('pg3').set('window', 'graphics');
model.result('pg3').set('windowtitle', '');
model.result.export('plot1').set('filename', filename);
model.result.export('plot1').run;

model.hist.disable;

t2 = toc;
fprintf('%d cost: %fs\n', i, t2);
end







