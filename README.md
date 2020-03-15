# bbox-augmentation
Program to Augment images with accompanying bbox data
The program takes in an xls file specifying
the file paths xmin,ymin, xmax, and ymax in the order
it then determines the limits of how one can augment the image via, zoom, rotation
and shift and then augments the file in such a way as have the boundaries of the box leave the image
it then determines a new xmin,ymin, xmax and ymax from the transformed image and saves both the image file 
and an excel file specifying the new points.
