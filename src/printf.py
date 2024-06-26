from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

# This file contains some accessory functions for format-printing.

def printf(output_file, str2print, *args):
    """
    Format-print the requested str2printing to a given output file 
    """
    print (str2print % args, end='', file = output_file, flush = True)

def printarFp (output_file, ar):
    """
    Format-print the input array ar of floating-point to a given output file.
    The array is printed without commas or newlined inside, and with a newline in the end.
    E.g.: 
    [1 2 3]
    
    """
    for item in ar:
        printf (output_file, '{:.5f}, ' .format(item))
    printf (output_file, '\n')

def printar (output_file, ar):
    """
    Format-print the input array ar to a given output file.
    The array is printed without commas or newlined inside, and with a newline in the end.
    E.g.: 
    [1 2 3]
    
    """
    ar=np.array(ar)
    printf (output_file, '{}\n' .format(str(ar).replace('\n', '')))

def printmat (output_file, mat, my_precision=0):
    """
    Format-print a given matrix to a given output file, using the requested precision (number of digits beyond the decimal point).
    """
    precision_str = '{{:.{}f}}\t' .format (my_precision)
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            printf (output_file, precision_str .format (mat[row][col]))
        printf (output_file, '\n')
    printf (output_file, '\n')

def invert_mat_bottom_up (mat):
    """
    Swap the matrix upside-down. 
    This is sometimes usefuly, because , we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
    """ 
    inverted_mat = np.empty (mat.shape)
    for i in range (mat.shape[0]):
        inverted_mat[i][:] = mat[mat.shape[0]-1-i][:]
    return inverted_mat        

def printFigToPdf (output_file_name):
    """
    Print the current fig to a PDF file
    """
    
    plt.savefig ('../res/{}.pdf' .format (output_file_name), bbox_inches='tight')


def printBinVec (binVec, grp=4):
    """
    format-print a binary-vec. Bits are grouped into grp-sized groups.
    Example:  
    > printBinVec ("000011110", 4)
    Will print:
    0000 1111 0
    """
    L=[]
    for i,b in enumerate(binVec):
        if b=="1":
            L.append("1")
        else: 
            L.append("0")
        if (i+1)%grp ==0 :
            L.append(" ")
    print ('{}' .format ("".join(L)))
        
def printfDict (outputFile, dict2print):
    """
    format-print a dictionary to a file
    """
    for key in dict2print:
        # printf (outputFile, f'{key} : {dict2print[key]}\n')
        # print (f'{key} : {dict2print[key]}\n', end='', file = outputFile, flush = True)
        # print (dict2print[key])
        printf (outputFile, f'{key} : {dict2print[key]}\n')
    printf (outputFile, '\n\n')
    
# debugFile = open ('../res/debug.txt', 'w')
# dict2print = {'gamad' : 1, 'nanas' : 2}
# printfDict (debugFile, dict2print)
