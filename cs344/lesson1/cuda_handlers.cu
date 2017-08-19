


__global__ void kernel_rgb2grey( uchar4 *d_rgbaImage,
                                 unsigned char *d_greyImage,
                                 int numRows, int numCols )
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x < numCols &&
         y < numRows )
    {
        int idxPix = x + y * numCols;
        d_greyImage[idxPix] = 0.299f * d_rgbaImage[idxPix].x +
                              0.587f * d_rgbaImage[idxPix].y +
                              0.114f * d_rgbaImage[idxPix].z;
    }
}






void rgba_to_greyscale( const uchar4* h_rgbaImage,
                        uchar4* d_rgbaImage,
                        unsigned char* d_greyImage,
                        size_t nRows, size_t nCols )
{

    // Let's work in blocks of 10x10 in a grid of nRows/10 x nCols/10
    const dim3 _blockSize( 10, 10, 1 );
    const dim3 _gridSize( nCols / 10, nRows / 10, 1 );

    kernel_rgb2grey<<<_gridSize,_blockSize>>>( d_rgbaImage, d_greyImage, nRows, nCols );
}
