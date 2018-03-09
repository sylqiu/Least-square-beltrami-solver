function uv_new = lsqc_solver(face,vertex,mu,landmark,target)
% lsqc_solver computes the solution of the Beltrami equation using the
% method described in 
% Parametrizing flat-foldable surfaces with incomplete data. Qiu et el.
% When using, provide at least two constraints.
% Inputs:
% face: Nx3 input mesh triangles
% vertex: Mx2 input mesh vertex coordinates
% mu: Nx1 complex, Beltrami coefficients defined on triangles
% landmark: index of the landmarked verteices 
% target: the constrained vertices target positions
%
% Outputs:
% uv_new: vertex coordinates of the solution mesh
%
% Copyright 2018: Qiu Di (sylvesterqiu@gmail.com)
landmark = [landmark;size(vertex,1)+landmark];
target = reshape(target,size(target,1)*2,1);
L2 = div_PtP_grad(mu,face,vertex);
A = unsigned_area_matrix(vertex,face,mu);
L = blkdiag(L2,L2);
M =L-2*A;
b = -M(:,landmark)*target;
b(landmark,:) = target;
M(landmark,:) = 0; 
M(:,landmark) = 0; 
M = M + sparse(landmark,landmark,ones(length(landmark),1), size(M,1), size(M,2));
uv_new = M\b;
uv_new = reshape(uv_new,size(uv_new,1)/2,2);
end

function mA = unsigned_area_matrix(vertex,face,mu)
% a = (1-2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
% b = -2*imag(mu)./(1.0-abs(mu).^2);
% g = (1+2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
a = 1*ones(size(mu,1),1);
b = a*0;
a(abs(mu)>1) = -a(abs(mu)>1);
g= a;
fi = face(:,1);
fj = face(:,2);
fk = face(:,3);
uxv0 = vertex(fj,2) - vertex(fk,2);
uyv0 = vertex(fk,1) - vertex(fj,1);
uxv1 = vertex(fk,2) - vertex(fi,2);
uyv1 = vertex(fi,1) - vertex(fk,1); 
uxv2 = vertex(fi,2) - vertex(fj,2);
uyv2 = vertex(fj,1) - vertex(fi,1);
%lengths
l = [sqrt(sum(uxv0.^2 + uyv0.^2,2)) ...
    sqrt(sum(uxv1.^2 + uyv1.^2,2)) ...
    sqrt(sum(uxv2.^2 + uyv2.^2,2))];
%half the perimeter of each triangle, denoted s
s = sum(l,2)*0.5;
%area of triangle with sides (a,b,c) = sqrt(s(s-a)(s-b)(s-c))
area = sqrt( s.*(s-l(:,1)).*(s-l(:,2)).*(s-l(:,3)));
area = sqrt(area);
A = uxv0./(2*area);
B = uxv1./(2*area);
C = uxv2./(2*area);
D = uyv0./(2*area);
E = uyv1./(2*area);
F = uyv2./(2*area);

q11 = A.*(A.*b-D.*a)+D.*(A.*g-D.*b);
q21 = A.*(B.*b-E.*a)+D.*(B.*g-E.*b);
q31 = A.*(C.*b-F.*a)+D.*(C.*g-F.*b);
q12 = B.*(A.*b-D.*a)+E.*(A.*g-D.*b);
q22 = B.*(B.*b-E.*a)+E.*(B.*g-E.*b);
q32 = B.*(C.*b-F.*a)+E.*(C.*g-F.*b);
q13 = C.*(A.*b-D.*a)+F.*(A.*g-D.*b);
q23 = C.*(B.*b-E.*a)+F.*(B.*g-E.*b);
q33 = C.*(C.*b-F.*a)+F.*(C.*g-F.*b);

N = size(vertex,1);
II = [fi;
      fj;
      fk;
      fi;
      fj;
      fi;
      fk;
      fj;
      fk;
      N+fi;
      N+fj;
      N+fk;
      N+fi;
      N+fj;
      N+fi;
      N+fk;
      N+fj;
      N+fk;
      ];
JJ = [N+fi;
      N+fj;
      N+fk;
      N+fj;
      N+fi;
      N+fk;
      N+fi;
      N+fk;
      N+fj;
      fi;
      fj;
      fk;
      fj;
      fi;
      fk;
      fi;
      fk;
      fj;
      ];
QQ = [q11;q22;q33;q12;q21;q13;q31;q23;q32;
    -q11;-q22;-q33;-q12;-q21;-q13;-q31;-q23;-q32];
mA = sparse(II,JJ,0.5*QQ);
end

function A = div_PtP_grad(mu,face,vertex)% this is the same with the generalized laplacian if mu<1
aaf = -(1-2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);
bbf = 2*imag(mu)./(1.0-abs(mu).^2);
ggf = -(1+2*real(mu)+abs(mu).^2)./(1.0-abs(mu).^2);

aaf(abs(mu)>1) = -aaf(abs(mu)>1);
bbf(abs(mu)>1) = -bbf(abs(mu)>1);
ggf(abs(mu)>1) = -ggf(abs(mu)>1);

af = aaf;
bf = bbf;
gf = ggf;
f0 = face(:,1);
f1 = face(:,2);
f2 = face(:,3);

uxv0 = vertex(f1,2) - vertex(f2,2);
uyv0 = vertex(f2,1) - vertex(f1,1);
uxv1 = vertex(f2,2) - vertex(f0,2);
uyv1 = vertex(f0,1) - vertex(f2,1); 
uxv2 = vertex(f0,2) - vertex(f1,2);
uyv2 = vertex(f1,1) - vertex(f0,1);
%lengths
l = [sqrt(sum(uxv0.^2 + uyv0.^2,2)) ...
    sqrt(sum(uxv1.^2 + uyv1.^2,2)) ...
    sqrt(sum(uxv2.^2 + uyv2.^2,2))];
%half the perimeter of each triangle, denoted s
s = sum(l,2)*0.5;
%area of triangle with sides (a,b,c) = sqrt(s(s-a)(s-b)(s-c))
area = sqrt( s.*(s-l(:,1)).*(s-l(:,2)).*(s-l(:,3)));
%
v00 = (af.*uxv0.*uxv0 + 2*bf.*uxv0.*uyv0 + gf.*uyv0.*uyv0)./area;
v11 = (af.*uxv1.*uxv1 + 2*bf.*uxv1.*uyv1 + gf.*uyv1.*uyv1)./area;
v22 = (af.*uxv2.*uxv2 + 2*bf.*uxv2.*uyv2 + gf.*uyv2.*uyv2)./area;

v01 = (af.*uxv1.*uxv0 + bf.*uxv1.*uyv0 + bf.*uxv0.*uyv1 + gf.*uyv1.*uyv0)./area;
v12 = (af.*uxv2.*uxv1 + bf.*uxv2.*uyv1 + bf.*uxv1.*uyv2 + gf.*uyv2.*uyv1)./area;
v20 = (af.*uxv0.*uxv2 + bf.*uxv0.*uyv2 + bf.*uxv2.*uyv0 + gf.*uyv0.*uyv2)./area;

I = [f0;f1;f2;f0;f1;f1;f2;f2;f0];
J = [f0;f1;f2;f1;f0;f2;f1;f0;f2];
V = [v00;v11;v22;v01;v01;v12;v12;v20;v20]./2;
%I,J are three copy of vertices, A is the matrix where the slot  
% (vertex X vertex) = (i,j) contains 
A = sparse(I,J,-V/2);

end