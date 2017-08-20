

__kernel void kernel_rgb2grey( __global uchar4* d_rgbaImage,
                               __global uchar* d_greyImage,
                               int numRows, int numCols  )
{

    int x = get_local_id( 0 ) + get_local_size( 0 ) * get_group_id( 0 );
    int y = get_local_id( 1 ) + get_local_size( 1 ) * get_group_id( 1 );

    if ( x < numCols &&
         y < numRows )
    {
        int idxPix = x + y * numCols;
        d_greyImage[idxPix] = 0.299 * d_rgbaImage[idxPix].x +
                              0.587 * d_rgbaImage[idxPix].y +
                              0.114 * d_rgbaImage[idxPix].z;
        
    }
}
