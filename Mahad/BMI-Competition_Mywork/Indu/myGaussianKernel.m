function kernel = myGaussianKernel(sigma, kernel_length)
%MYGAUSSIANKERNEL Creates a 1-D Gaussian kernel.
%
%   kernel = myGaussianKernel(sigma, kernel_length) returns a row vector
%   representing a Gaussian kernel with standard deviation sigma and the 
%   specified kernel_length. If kernel_length is not provided, it defaults to 
%   round(5*sigma).
%
%   The kernel is normalized so that its elements sum to 1.

    if nargin < 2
        kernel_length = round(5 * sigma);
    end
    half_len = (kernel_length - 1) / 2;
    x = -half_len:half_len;
    kernel = exp(-x.^2/(2*sigma^2));
    kernel = kernel / sum(kernel);
    kernel = reshape(kernel, 1, []); % Ensure row vector
end
