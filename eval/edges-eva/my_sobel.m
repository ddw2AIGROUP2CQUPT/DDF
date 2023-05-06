function [o,ot]=my_sobel(i,k,t)
kt=k';
if length(size(i))>2
    i= rgb2gray(i);
end
i=double(i);
r=size(i,1);
c=size(i,2);
m=zeros([r+2,c+2]);
I=m;
I(2:end-1,2:end-1)=i;
m=zeros([r,c]);
ot=m;
for x=1:r-2
    for y=1:c-2
        s1=sum(sum(k.*I(x:x+2,y:y+2)));
        s2=sum(sum(kt.*I(x:x+2,y:y+2)));
        
        ot(x,y)=s1;
        m(x,y)=sqrt(s1.^2+s2^2);
    end
end

o=max(m,t);
o(o==round(t))=0;
ot=uint8(ot);
o=uint8(o);
o=imcomplement(o);
end