function[ERPdata, flash_id] =  makeDataset(subject)
  % input 
  %   subject: (strings) mat file name 
  % output
  %   ERPdata:3D array (trial, ch, sample)
  %   flash_id: 2D array (trial,2). 1st column means flash image number (1-4). 2nd column means target or not (1/0).
  %
  
  load([subject '.mat'])

  %chNum = size(dataset,2);
  chNum = 9;
  chIdx = 1:chNum;
  trials = size(dataset,3);
  flashs = size(flashImageNumber, 2);
  
  %resamplingRate = samplingRate;
  resamplingRate = 20;
  dataLength = 0.7; % [s]
  baselineLength = 0.1; % [s]
  
  
  
  %filtering
  %filterFreq = [0.1, 30];
  %w = filterFreq ./ (0.5 * samplingRate);
  %filterOrder = 3;
  %[B, A] = butter(filterOrder, w);
  %dataset = filtfilt(B, A, dataset);
  
  
  %%make ERPdata
  tempBaseline = zeros(baselineLength*samplingRate, chNum, trials*flashs);
  tempERPdata = zeros(dataLength*samplingRate, chNum, trials*flashs);
  ERPdata = zeros(dataLength*resamplingRate, chNum, trials*flashs);
  % split data
  for ii = 1:trials
    for jj = 1:flashs
      idx = (ii-1)*flashs;
      tempBaseline(:,:,idx+jj) = dataset(flashTimingSet(ii,jj)-baselineLength*samplingRate+1 :flashTimingSet(ii,jj),chIdx,ii);
      tempERPdata(:,:,idx+jj) = dataset(flashTimingSet(ii,jj) :flashTimingSet(ii,jj)+ dataLength*samplingRate-1,chIdx,ii);
    end
  end
  
  
  % arrange baseline
  tempERPdata = tempERPdata-repmat(mean(tempBaseline,1),[dataLength*samplingRate,1,1]);
  
  %resample
  if resamplingRate == samplingRate
    ERPdata = tempERPdata;
  else
    resampleRatio = floor(samplingRate/resamplingRate);
    for ii = 1:dataLength*resamplingRate
      idx = (ii-1)*resampleRatio+1:ii*resampleRatio;
      ERPdata(ii,:,:) = mean(tempERPdata(idx,:,:));
    end
  
  end
  ERPdata = permute(ERPdata, [3 2 1]);
  
  %%make flash_id
  flash_id = zeros(trials*flashs, 2);
  for ii = 1:trials
    index = (ii-1)*flashs + 1:(ii-1)*flashs + flashs;
    flash_id(index, 1) = flashImageNumber(ii,:)';
    flash_id(index, 2) = (flashImageNumber(ii,:)==target(ii))'; 
  end

 
end
