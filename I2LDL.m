function [W,Q] = I2LDL(features,obrT,labels,lambda_1,lambda_2,lambda_3,rho_1,rho_2)
%ADMMMAXIDE Summary of this function goes here
%   This code implements the I2LDL method
%   Input
%       -features: n*d features in learning
%       -obrT: n*m 0/1 matrix, 1 means the corresponding position in labels
%              is observed
%       -labels:  n*m ground truth labels
%       -lambda: regularization parameter
%       -rho: regularization parameter
%   Output
%       W d*m weight matrix for low-rank part
%       Q d*m weight matrix for sparse part
%       iterations

[n,m]=size(labels);
d=size(features,2);
k=min(d,m);
max_iter=50;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);

epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);

%need to judge
epsilon_abs=1e-4;
epsilon_rel=1e-2;

U=rand(d,k);
V=rand(k,m);
Q=rand(d,m);
Z=rand(n,m);
G=rand(n,m);
Y_1=rand(n,m);
Y_2=rand(n,m);

t=0;
while(t<max_iter)
    t=t+1;
    disp(['Enter iteration ' num2str(t)]);
    Z0=Z;
    G0=G;

    %updateQ:
    M=zeros(n,m);
    
    for i=1:n
        m=length(labels(i,:));
        unobrT=ones(1,m);
        unobrT=unobrT-obrT(i,:);
        [p,q]=size(Z(i,:));
        
        H=(1+rho_2)*obrT(i,:)+rho_2*unobrT;
        H=diag(H);
        
        H= double(H)+2*lambda_2*(pinv(features(i,:))'*pinv(features(i,:)));
        
        f=(Y_2(i,:)-(labels(i,:).*obrT(i,:)-Z(i,:))-rho_2*G(i,:)).*obrT(i,:)+(Y_2(i,:)-rho_2*G(i,:)).*unobrT;
        f=f';
        
        A=-eye(m);
        b=zeros(p,q);
        Aeq=ones(m,1)';
        beq=1-Z(i,:)*Aeq';
        
        options=optimoptions(@quadprog,'Display','off');
        f = double(f);
        x=quadprog(H,f,A,b,Aeq,beq,[],[],[],options);
        if size(x,1)==0
            x=zeros(1,m);
        end
        M(i,:)=x';
    end
    Q=pinv(features'*features)*(features'*M);

    %updateZ:
    newZ = zeros(n, m);
    M = features * U * V;
    N = features * Q;

    for i = 1:n
        [p,q]=size(N(i, :));

        unobrT=ones(1,m);
        unobrT=unobrT-obrT(i, :);
        
        H=(1+rho_1)*obrT(i, :)+rho_1*unobrT;
        H=diag(H);
        
        f=(Y_1(i, :)-(labels(i, :).*obrT(i, :)-N(i, :))-rho_1*M(i, :)).*obrT(i, :)+(Y_1(i, :)-rho_1*M(i, :)).*unobrT;
        f=f';
        
        A=-eye(m);
        b=zeros(p,q);
        
        Aeq=ones(m,1)';
        beq=1-N(i, :)*Aeq';
        
        beq = double(beq);
        
        options=optimoptions(@quadprog,'Display','off');
        f = double(f);
        x=quadprog(H,f,A,b,Aeq,beq,[],[],[],options);
        if size(x,1)==0
            x=zeros(1,m);
        end
        
        newZ(i,:)=x';
    end
    Z = newZ;

    %updateU:
    A=(rho_1/(2*lambda_1))*(features'*features);
    B=pinv(V*V');
    C=(1/(2*lambda_1))*(features'*rho_1*Z*V'-features'*Y_1*V')*pinv(V*V');
    U=sylvester(A,B,C);
    
    %updateV:
    V=(pinv(2*lambda_1+rho_1*U'*(features'*features)*U))*(rho_1*U'*features'*Z-U'*features'*Y_1);
    
    %updateG:
    A = features*Q-Y_2/rho_2;
    lambda = lambda_3/rho_2;
    G = sign(A) .* max(abs(A) - lambda, 0);

    Y_1=Y_1+rho_1*(features*U*V-Z);
    Y_2=Y_2+rho_2*(features*Q-G);
    rho_1 = max(50,rho_1*1.1);
    rho_1 = max(50,rho_1*1.1);

    %primal residual
    convergence1(t,1)=norm(features*U*V-Z,'fro')+norm(features*Q-G,'fro');
    
    %dual residual
    convergence2(t,1)=norm(rho_1*features'*(Z-Z0),'fro')+norm(rho_2*features'*(G-G0),'fro');
    
    %primal epsilon
    epsilon_primal_Z(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(features*U*V,'fro'),norm(Z,'fro'));
    epsilon_primal_G(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(features*Q,'fro'),norm(G,'fro'));
    epsilon_primal(t,1)=epsilon_primal_Z(t,1)+epsilon_primal_G(t,1);
    %dual epsilon
    epsilon_dual_Z(t,1)=sqrt(d)*epsilon_abs+epsilon_rel*norm(features'*Y_1,'fro');
    epsilon_dual_G(t,1)=sqrt(d)*epsilon_abs+epsilon_rel*norm(features'*Y_2,'fro');
    epsilon_dual(t,1)=epsilon_dual_Z(t,1)+epsilon_dual_G(t,1);
    
    
    if (convergence1(t,1)<=epsilon_primal(t,1) && convergence2(t,1)<=epsilon_dual(t,1))
        break;
    end
end

W = U*V;

end

