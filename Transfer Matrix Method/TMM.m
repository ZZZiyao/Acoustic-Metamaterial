% 定义材料参数
rho1 = 2730;  %铝的密度(kg/m^3)
c1 = 3242;  %铝的剪切波波速(m/s)
rho2 = 1180;  %环氧树脂的密度(kg/m^3)
c2= 1161;  %环氧树脂的剪切波波速(kg/m^3)

%定义几何尺寸
a1 = 0.75;  %
a2 = 0.75;
a = a1+a2;

%定义入射角
theta = 0.5;

%建立频率数组
f = linspace(0,20000,50);

%计算波数
alpha_z = 2*pi*f/c1*sin(theta);
alpha_x1 = sqrt((2*pi*f/c1).^2-alpha_z.^2);
alpha_x2 = sqrt((2*pi*f/c2).^2-alpha_z.^2);
%计算F
F = alpha_x1*rho1*c1^2/(alpha_x2*rho2*c2^2);

%计算色散曲线控制方程右侧
coska = cosh(alpha_x1*a1*i).*cosh(alpha_x2*a2*i)+0.5*(F+1/F).*sinh(alpha_x1*a1*i).*sinh(alpha_x2*a2*i);

%计算Bloch波矢
k = acos(coska)/a;

%画出色散曲线图
subplot(2,1,1);
plot(real(k*a/pi),f, 'LineWidth', 4);
title('色散曲线实部');
xlabel('k (pi/a)');
ylabel('频率 (Hz)');

subplot(2,1,2);
plot(abs(imag(k*a/pi)),f, 'LineWidth', 4);
title('色散曲线虚部');
xlabel('k (pi/a)');
ylabel('频率 (Hz)');




