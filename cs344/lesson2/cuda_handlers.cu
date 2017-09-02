

__global__
void kernel_gaussian_blur( unsigned char* d_in,
                           unsigned char* d_out,
                           int numRows, int numCols,
                           float* d_filter, 
                           int filterDim )
{
    // NOTE: Be sure to compute any intermediate results in floating point
    // before storing the final result as unsigned char.

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x >= numCols ||
         y >= numRows )
    {
        return;
    }
    
    // x, y are the coordinates of the center pixel

    // Apply the filter
    float mask_sum = 0.0;

    for ( int dx = -filterDim / 2; dx <= filterDim / 2; dx++ )
    {
        for ( int dy = -filterDim / 2; dy <= filterDim / 2; dy++ )
        {
            int xx = min( max( x + dx, 0 ), numCols - 1 );
            int yy = min( max( y + dy, 0 ), numRows - 1 );

            float filter_val = d_filter[ ( dy + filterDim / 2 ) * filterDim +
                                         ( dx + filterDim / 2 ) ];
            float pix_val = ( float ) ( d_in[ yy * numCols + xx ] );

            mask_sum += pix_val * filter_val;
        }
    }

    d_out[ y * numCols + x ] = ( int ) mask_sum;

}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void kernel_separateChannels( uchar4* d_inputImageRGBA,
                              int numRows,
                              int numCols,
                              unsigned char* d_redChannel,
                              unsigned char* d_greenChannel,
                              unsigned char* d_blueChannel )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x >= numCols ||
         y >= numRows )
    {
        return;
    }
    int idxPix = x + y * numCols;

    d_redChannel[idxPix]    = d_inputImageRGBA[idxPix].x;
    d_greenChannel[idxPix]  = d_inputImageRGBA[idxPix].y;
    d_blueChannel[idxPix]   = d_inputImageRGBA[idxPix].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void kernel_recombineChannels( const unsigned char* const d_redChannel,
                               const unsigned char* const d_greenChannel,
                               const unsigned char* const d_blueChannel,
                               uchar4* const d_outputImageRGBA,
                               int numRows,
                               int numCols )
{
    const int2 _thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y );

    const int _thread_1D_pos = _thread_2D_pos.y * numCols + _thread_2D_pos.x;

    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if ( _thread_2D_pos.x >= numCols || _thread_2D_pos.y >= numRows )
    {
        return;
    }

    unsigned char _red   = d_redChannel[_thread_1D_pos];
    unsigned char _green = d_greenChannel[_thread_1D_pos];
    unsigned char _blue  = d_blueChannel[_thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 _outputPixel = make_uchar4( _red, _green, _blue, 255 );

    d_outputImageRGBA[_thread_1D_pos] = _outputPixel;
}

void apply_gaussian_blur( uchar4* h_inputImageRGBA, 
                          uchar4* d_inputImageRGBA,
                          uchar4* d_outputImageRGBA, 
                          size_t numRows, size_t numCols,
                          unsigned char* d_red, 
                          unsigned char* d_green, 
                          unsigned char* d_blue,
                          unsigned char* d_redBlurred, 
                          unsigned char* d_greenBlurred, 
                          unsigned char* d_blueBlurred,
                          float* d_filter,
                          int filterDim )
{
    // Set reasonable block size (i.e., number of threads per block)
    const dim3 blockSize( 10, 10 );

    // Compute correct grid size (i.e., number of blocks per kernel launch) ...
    // from the image size and and block size.
    const dim3 gridSize( numCols / 10, numRows / 10 );

    // Launch a kernel for separating the RGBA image into different color channels
    kernel_separateChannels<<<gridSize, blockSize>>>( d_inputImageRGBA, 
                                                      numRows, numCols, 
                                                      d_red, d_green, d_blue );
    cudaDeviceSynchronize();

    // Call your convolution kernel here 3 times, once for each color channel.
    kernel_gaussian_blur<<<gridSize, blockSize>>>( d_red, d_redBlurred,
                                                   numRows, numCols, 
                                                   d_filter, filterDim );
    kernel_gaussian_blur<<<gridSize, blockSize>>>( d_green, d_greenBlurred,
                                                   numRows, numCols, 
                                                   d_filter, filterDim );
    kernel_gaussian_blur<<<gridSize, blockSize>>>( d_blue, d_blueBlurred,
                                                   numRows, numCols, 
                                                   d_filter, filterDim );

    cudaDeviceSynchronize();

    // Now we recombine your results. We take care of launching this kernel for you.
    //
    // NOTE: This kernel launch depends on the gridSize and blockSize variables,
    // which you must set yourself.
    kernel_recombineChannels<<<gridSize, blockSize>>>( d_redBlurred,
                                                       d_greenBlurred,
                                                       d_blueBlurred,
                                                       d_outputImageRGBA,
                                                       numRows,
                                                       numCols );
    cudaDeviceSynchronize();

}