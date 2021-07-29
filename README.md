# volume_compare
The compV3.h5 is used to determine of which two sample photo has the higher volume. It can correctly identify the higher volume about 87% of the time using the 'compare()' function. <br />

The sorted.txt has a list of the 100 images in the sorted folder in order from smallest to largest. The 'find_position()' function identifies where in the sorted list a new image should be inserted. This index in the list (divided by 10) is returned to the user as the detected volume. i.e: The function will return a value from 0.0 to 10.0 depending on where in the length 100 list the image should be inserted (but it is not actually inserted). <br /> 

The algorithm for finding the index is a binary search and it has subpar reliability of finding the volume. Accuracy may be increased if another algorithm is used to find the position using the compare() function. 
