%读取参数变量列表
params = xlsread('parameters.xlsx', 'sheet1');
Es = params(:, 1)*1e6;
Ps = params(:, 2);
rhos = params(:, 3)*1e3;
Er = params(:, 4)*1e6;
Pr = params(:, 5);
rhor = params(:, 6)*1e3;
L1 = params(:, 7);
L2 = params(:, 8);
L3 = params(:, 9);
L4 = params(:, 10);

%以列表顺序开始计算
for i=1:100 %设置计算范围区间
tic; %开始计时

filename = ['dispersion_curves/', num2str(i), '.csv'];

model=mphload('P');

model.param.set('Es', num2str(Es(i)));
model.param.set('Ps', num2str(Ps(i)));
model.param.set('rhos', num2str(rhos(i)));
model.param.set('Er', num2str(Er(i)));
model.param.set('Pr', num2str(Pr(i)));
model.param.set('rhor', num2str(rhor(i)));
model.param.set('L1', num2str(L1(i)));
model.param.set('L2', num2str(L2(i)));
model.param.set('L3', num2str(L3(i)));
model.param.set('L4', num2str(L4(i)));

model.study('std1').run;

model.result.export.create('plot1', 'Plot');
model.result.export('plot1').set('plotgroup', 'pg3');
model.result.export('plot1').set('filename', filename);
model.result.export('plot1').run

%清楚计算历史
model.hist.disable;

t2 = toc;
fprintf('%d   cost: %fs\n', i, t2);

end
