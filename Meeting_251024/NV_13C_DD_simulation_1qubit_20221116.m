clear all
    tic;
    % updated in 07/12/2022 by Paul Junghyun Lee
    
    %% Define dimension, pauli matrices

    i   = sqrt(-1);
    sx  = 1/sqrt(2)*[0, 1, 0; 1, 0, 1; 0, 1, 0];
    sy  = 1/sqrt(2)/i*[0, 1, 0; -1, 0, 1; 0, -1, 0];
    sz  = [1, 0, 0; 0, 0, 0; 0, 0, -1];
    %sz  = [1, 0, 0; 0, -1, 0; 0, 0, 0];
    I   = [1, 0, 0; 0, 1, 0; 0, 0, 1];
    
    % Rotation matrix projected into 2 level system
    Sxp  = [0, 1, 0; 1, 0, 0; 0, 0, 0];
    Sxm  = [0, 0, 0; 0, 0, 1; 0, 1, 0];
    
    Syp  = 1/i*[0, 1, 0; -1, 0, 0; 0, 0, 0];
    Sym  = 1/i*[0, 0, 0; 0, 0, 1; 0, -1, 0];
    
    % Pauli basis for 13C nuclear spin
    Ix  = 1/2*[0, 0, 1; 0, 0, 0; 1, 0, 0];    
    Iy  = 1/2*1/i*[0, 0, 1; 0, 0, 0; -1, 0, 0];
    Iz  = 1/2*[1, 0, 0; 0, 0, 0; 0, 0, -1];
    
    %% Define sweep parameters
    Sweep = 5001;  
    N     = Sweep;
    B     = 440.1;%448.84; %[G] magnetic field
    
    % 13C nuclear spin parameters
    gammaN = 2*pi*1.0705e-3; %[MHz/G]
    Al    = 2*pi*-0.073; %[MHz] % A_|| hyperfine term (Opposite Sign)
    Ap    = 2*pi*0.042; %[MHz] % A_per hyperfine term


    ohm   = 50E-3; %[MHz]
    w0 = B*gammaN;

    w1 =sqrt((Ap*Ap) + ((w0-Al)*(w0-Al)));
    w2 = sqrt((Ap*Ap) + ((w0+Al)*(w0+Al)));

    
    T     = 28; % sweep tau [us]
    t = linspace(0,T,N);
    freq = 2*pi*linspace(300E-3, 700E-3, N); %[MHz]
    n = 16; % number of pi pulses

  

    
    %% Define gate operations
    
    % Single Q ms=+1
    U090xp = UO(1,0,pi/4,0,0);
    U090xmp = UO(1,0,-pi/4,0,0);
    U090yp = UO(1,0,pi/4,pi/2,0);
    U090ymp = UO(1,0,-pi/4,pi/2,0);
    U180xp = UO(1,0,pi/2,0,0);
    U180xmp = UO(1,0,-pi/2,0,0);
    U180yp = UO(1,0,pi/4,pi/2,0);
    U180ymp = UO(1,0,-pi/4,pi/2,0);
    
    % Single Q ms=-1
    U090xm = UO(0,1,pi/4,0,0);
    U090xmm = UO(0,1,-pi/4,0,0);
    U180xm = UO(0,1,pi/2,0,0);
    U180xmm = UO(0,1,-pi/2,0,0);
    
    %% Define initial state of the system
    irho_p = [1,0,0;0,0,0;0,0,0];%;0,0,0;0,0,0];
    irho_m = [0,0,0;0,0,0;0,0,1];%0,0,0;0,0,1];
    irho_z = [0,0,0;0,1,0;0,0,0];%0,1,0;0,0,0];
    irho_mix = [1/2,0,0;0,1/2,0;0,0,0];
    irho_Z = [0,0,0;0,0,0;0,0,1];%
    irho_MIX = [1/2,0,0;0,0,0;0,0,1/2];
    
    %irho = kron(irho_z,irho_MIX);
    irho = kron(irho_z,irho_MIX);
    
    % Initialization
    rho_0 = kron(U090xm,I)*irho*(kron(U090xm,I))'; % superposition state on NV
    
    
    
for h=1:N
       
        %w1 = B*gammaN - w;
        %w2 = (w0+w1)/2;
        w3 = (w2+w1)/2;
        ham = Al*kron(sz,Iz) + Ap*kron(sz,Ix) + B*gammaN*kron(I,Iz);
        %ham = (w0-freq(h))*kron(irho_z, Iz) + (w1-freq(h))*kron(irho_m, Iz) + ohm*kron(I, Ix);
        

        [eigvecs, eigvals] = eig(full(ham));  % diagonalizing the Hamiltonian
        E = eigvals;                         % exponent of eigenvalues
        U_H= eigvecs';                      % unitary matrix formed by eigenvectors
        
        %free evolution unitary operator
        %t(h) = 2*4;
        %tau = t(h);
        U_e2 = U_H'*(expm(-i*E*t(h)/2)*U_H); % for tau/2
        U_e  = U_H'*(expm(-i*E*t(h))*U_H);  % for tau
        
        

        rho_1 = U_e2 * rho_0 * U_e2'; % first tau/2
        rho_1c = rho_1;

        %CPMG
         for k=1:n-1
             rho_1c = U_e * kron(U180xm,I) * rho_1c * kron(U180xm,I)' * U_e';
         end

        rho_2 = U_e2 * kron(U180xm,I) * rho_1c * kron(U180xm,I)' * U_e2';
        %rho_2 = U_e2 * rho_1 * U_e2';
        rho_3 = kron(U090xmm,I) * rho_2 * (kron(U090xmm,I))'; % last pi/2

        Sa(h) = real(trace(irho_z*PartialTrace(rho_3,2))); % NV state 0 population readout


end
data = xlsread('C:\Users\KIST\Desktop\Test\CPMG.xlsx');
figure;
plot(data(:, 1)*1E+6 + 0.06, data(:, 2));
hold on;
t0 = t/2;
plot(t0,Sa);
%hold on;
%plot(freq/(2*pi), Sa);

%wm = (w0 - Al/2)/(2*pi) - (2/(4.1*tau));
%{
axis([0 14 0 1]);
figure;
plot(t0, Sa_w);
figure;
plot(t0, Sa_e);
%}
% Finding pi/2 pulse nn for nuclear spin rotation
[a,b]=min(Sa)
tau = t(b); % minimum tau

nn = 40;
nn_r = 2*linspace(1,nn,nn);
irho = kron(irho_z,irho_MIX);
%irho = kron(irho_z,irho_p);
    
    % Initialization
    rho_0 = kron(U090xp,I)*irho*(kron(U090xp,I))'; % superposition state on NV

for h=1:nn
%free evolution unitary operator
        U_e2 = U_H'*(expm(-i*E*tau/2)*U_H); % for tau/2
        U_e  = U_H'*(expm(-i*E*tau)*U_H);  % for tau
        
        rho_1 = U_e2 * rho_0 * U_e2'; % first tau/2
        
        for k=1:2*h-1
            rho_1 = U_e * kron(U180xp,I) * rho_1 * kron(U180xp,I)' * U_e';
        end
        
        rho_2 = U_e2 * kron(U180xp,I) * rho_1 * kron(U180xp,I)' * U_e2';
        rho_3 = kron(U090xmp,I) * rho_2 * (kron(U090xmp,I))'; % last pi/2
  
        Sb(h) = real(trace(irho_z*PartialTrace(rho_3,2))); % NV state 0 population readout
       % Sb(h) = real(trace(sz*PartialTrace(rho_3,1))); % NV state 0 population readout
end   

 %figure;
 %plot(nn_r,Sb);

[a,b] = min(Sb)
if mod(b,2)==1
    b = b+1;
    tau = tau - tau/3.268/b
end
Ng = b % number of pi pulses to generate pi/2 x gate

%NU090 = NU0(Al,Ap,tau,Ng);
NU090 = NU0(Al,Ap,1.386*2,18);  %% CRx Tau, N fixed


frho = kron(U090xmp,I) * NU090 * kron(U090xp,I) * irho * kron(U090xp,I)' * NU090' * kron(U090xmp,I)';
real(trace(irho_z*PartialTrace(frho,2)))

%% Nuclear spin initialization

N     = 100;%1001;
T     = 100%2/Ng; % sweep tau [us]
t     = linspace(0,T,N);
t2     = linspace(0.8,1.0,N);
t0 = t2*2;
irho = kron(irho_z,irho_MIX);

% Initialization sequence

% first pi/2 y
rho_1 = kron(U090xp,I) * irho * kron(U090xp,I)';
%rho_1 = irho;%kron(U090xp,I) * irho * kron(U090xp,I)';

% conditional rotation on nuclear spin
rho_2 = NU090 * rho_1 * NU090';
rho_r = zeros(3,3,4);
count = 1;

for k=1:N
    % pi/2 x on NV and Rz pi/2 on nuclear spin
    %rho_3 = kron(U090xp,I) * rho_2 * kron(U090xp,I)';
    rho_3 = NU0(Al,Ap,t0(k),60) * rho_2 * NU0(Al,Ap,t0(k),60)';
    %rho_3 = NU0(Al,Ap,2.22*2,4) * rho_2 * NU0(Al,Ap,2.22*2,4)';
    %rho_3 = NU0(Al,Ap,t0,2*k) * rho_3 * NU0(Al,Ap,t0, 2*k)';
    %rho_3 = NU0(Al,Ap,2.22*2,4) * rho_3 * NU0(Al,Ap,2.22*2,4)';
    %rho_3 = NU0(Al,Ap,tau, k*2) * rho_1 * NU0(Al,Ap,tau,k*2)';
    
    % conditional rotation on nuclear spin
    rho_4 = NU090 * rho_3 * NU090';
    rho_4 = kron(U090xp,I) * rho_4 * kron(U090xp,I)';
    SN(k) = real(trace(irho_z*PartialTrace(rho_4,2)));
%     SNx(k) = real(trace(Ix*PartialTrace(rho_3,1)));
%     SNy(k) = real(trace(Iy*PartialTrace(rho_3,1)));
%     SNz(k) = real(trace(Iz*PartialTrace(rho_3,1)));
    
%     if SN(k) > 0.998
%         rho_r(:,:,count) = PartialTrace(rho_4,1);
%         count = count + 1;
%     end
end

    %% Plot result
   
     %figure; 
     %hold on
     %plot(t2,SN);
     %plot(t,SN);
    
    
%     plot(t,SNy);
%     plot(t,SNz);
    
    toc;
    
    %% Generating NV spin gate function
    
    function U = UO(B1,B2,a,D1,D2)
        
    i   = sqrt(-1);
    gamma = 2*pi*2.8; %[MHz/G]
    D     = 2870; %[MHz]
    %B1    = 2*pi*1/gamma; %[MHz]
    %B2    = 2*pi*1/gamma; %[MHz]
    %w_e   = gamma*sqrt(B1^2+B2^2)/2/sqrt(2);
%     D1    = 0;
%     D2    = 0;
    
        U = [(B2^2+B1^2*cos(a))/(B1^2+B2^2), -i*B1*exp(-i*D1)*sin(a)/sqrt(B1^2+B2^2), ((-1+cos(a))*B1*B2*exp(-i*(D1-D2)))/(B1^2+B2^2); ...
            -i*B1*exp(i*D1)*sin(a)/sqrt(B1^2+B2^2), cos(a), -i*B2*exp(i*D2)*sin(a)/sqrt(B1^2+B2^2); ...
            ((-1+cos(a))*B1*B2*exp(i*(D1-D2)))/(B1^2+B2^2), -i*B2*exp(-i*D2)*sin(a)/sqrt(B1^2+B2^2), (B1^2+B2^2*cos(a))/(B1^2+B2^2)];
    end
    
    %% Generating nuclear spin gate function
    
    function NU = NU0(Al,Ap,tau,Ng)
    
    i   = sqrt(-1);
    sx  = 1/sqrt(2)*[0, 1, 0; 1, 0, 1; 0, 1, 0];
    sy  = 1/sqrt(2)/i*[0, 1, 0; -1, 0, 1; 0, -1, 0];
    sz  = [1, 0, 0; 0, 0, 0; 0, 0, -1];
    %sz  = [1, 0, 0; 0, -1, 0; 0, 0, 0];
    I   = [1, 0, 0; 0, 1, 0; 0, 0, 1];
    
    % Rotation matrix projected into 2 level system
    Sxp  = [0, 1, 0; 1, 0, 0; 0, 0, 0];
    Sxm  = [0, 0, 0; 0, 0, 1; 0, 1, 0];
    
    Syp  = 1/i*[0, 1, 0; -1, 0, 0; 0, 0, 0];
    Sym  = 1/i*[0, 0, 0; 0, 0, 1; 0, -1, 0];
    
    % Pauli basis for 13C nuclear spin
    Ix  = 1/2*[0, 0, 1; 0, 0, 0; 1, 0, 0];    
    Iy  = 1/2*(-i)*[0, 0, 1; 0, 0, 0; -1, 0, 0];
    Iz  = 1/2*[1, 0, 0; 0, 0, 0; 0, 0, -1];
    
    % Single Q ms=+1
    U090xp = UO(1,0,pi/4,0,0);
    U090xmp = UO(1,0,-pi/4,0,0);
    U180xp = UO(1,0,pi/2,0,0);
    U180xmp = UO(1,0,-pi/2,0,0);
    
    % Single Q ms=-1
    U090xm = UO(0,1,pi/4,0,0);
    U090xmm = UO(0,1,-pi/4,0,0);
    U180xm = UO(0,1,pi/2,0,0);
    U180xmm = UO(0,1,pi/2,0,0);
    
    B     = 448.4; %[G] magnetic field
    gammaN = 2*pi*1.071e-3; %[MHz/G]
    NU = kron(I,I);
    
    ham = Al*kron(sz,Iz) + Ap*kron(sz,Ix) + B*gammaN*kron(I,Iz);
    [eigvecs, eigvals] = eig(full(ham));  % diagonalizing the Hamiltonian
    E = eigvals;                         % exponent of eigenvalues
    U_H= eigvecs';                      % unitary matrix formed by eigenvectors

    %free evolution unitary operator
    U_e  = U_H'*(expm(-i*E*tau)*U_H);  % for tau
    U_e2  = U_H'*(expm(-i*E*tau/2)*U_H);
    
        NU = U_e2 * NU;
    
        for h=1:Ng-1
            NU = U_e * kron(U180xp,I) * NU;
        end
        
        NU = U_e2 * kron(U180xp,I) * NU;
    end