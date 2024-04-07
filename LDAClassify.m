function [class, posterb] = LDAClassify(X,classmode)
% Standard Linear Discriminant Analysis (Classify)
% 
% X         : data to be classified with features in columns
% classmode : classifier parameters
% class     : predicted class
% posterb   : posterior probability or classification score
% 
% 
% Yu Zhang, RIKEN & ECUST, 2012.01.17
%

Me=classmode.mean;
invSn=classmode.invSn;
w=invSn*(Me(1,:)-Me(2,:))';
thresh=w'*mean(Me,1)';
posterb=w'*X';
posterb=posterb';
class=zeros(1,length(posterb));
class(posterb>thresh)=1;
class(posterb<thresh)=2;
