function [data,label,test] = h5_reader(path)
    
    h5_info=h5info(path);
   
    if length(h5_info.Datasets)==3
        data = h5read(path,'/data');
        label = h5read(path,'/label');
        test = h5read(path,'/test');
        data = permute(data,[4,3,2,1]);
        label = permute(label,[4,3,2,1]);
        test = permute(test,[4,3,2,1]);
    elseif length(h5_info.Datasets)==2
        data = h5read(path,'/data');
        data = permute(data,[4,3,2,1]);
        label = h5read(path,'/label');
        label = permute(label,[4,3,2,1]);
        test=0;
        
    elseif length(h5_info.Datasets)==1
        if length(h5_info.Datasets.Dataspace.Size)==4
            data = h5read(path,'/data');
            data = permute(data,[4,3,2,1]);
        elseif length(h5_info.Datasets.Dataspace.Size)==3
            data = h5read(path,'/data');
            data = permute(data,[3,2,1]);
        elseif length(h5_info.Datasets.Dataspace.Size)==2
            data = h5read(path,'/data');
            data = permute(data,[2,1]);
        else
            disp('Error reading data size');
        end
        
        label=0;
        test=0;
  
    end

end