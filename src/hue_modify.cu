#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <png.h>

// Funcao que aplica a matriz de transformacao A
// ao pixel px = (r, g, b)
// (new_r, new_g, new_b)' = A * (r, g, b)'
__host__ __device__ void modify_pixel(png_bytep px, double *A) {
    double r = px[0] / 255.0;
    double g = px[1] / 255.0;
    double b = px[2] / 255.0;

    double new_r = A[0] * r + A[1] * g + A[2] * b;
    double new_g = A[3] * r + A[4] * g + A[5] * b;
    double new_b = A[6] * r + A[7] * g + A[8] * b;

    new_r = fmin(fmax(new_r, 0.0), 1.0);
    new_g = fmin(fmax(new_g, 0.0), 1.0);
    new_b = fmin(fmax(new_b, 0.0), 1.0);

    px[0] = (png_byte) round(new_r * 255.0);
    px[1] = (png_byte) round(new_g * 255.0);
    px[2] = (png_byte) round(new_b * 255.0);
}

// Altera a matiz (hue) de uma imagem sequencialmente
void modify_hue_seq(png_bytep image, int width, int height, double hue_diff) {
    double c = cos(2 * M_PI * hue_diff);
    double s = sin(2 * M_PI * hue_diff);
    double one_third = 1.0 / 3.0;
    double sqrt_third = sqrt(one_third);

    // Matriz A compoe as operacoes de
    // conversao de RGB para HSV, mudanca de hue,
    // e conversao de HSV de volta para RGB
    // (new_r, new_g, new_b)' = A * (r, g, b)'
    // https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color

    double a11 = c + one_third * (1.0 - c);
    double a12 = one_third * (1.0 - c) - sqrt_third * s;
    double a13 = one_third * (1.0 - c) + sqrt_third * s;
    double a21 = a13; double a22 = a11; double a23 = a12;
    double a31 = a12; double a32 = a13; double a33 = a11;

    double A[9] = {a11, a12, a13, a21, a22, a23, a31, a32, a33};

    for (int i = 0; i < height; i++) {
        png_bytep row = &(image[i * width * 3]);
        for (int j = 0; j < width; j++) {
            png_bytep px = &(row[j * 3]);
            modify_pixel(px, A);
        }
    }
}

// Funcao auxiliar para identificar erros CUDA
void checkErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s [Erro CUDA: %s]\n",
                msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Kernel CUDA para alteracao do hue
// Voce deve modificar essa funcao no EP3
__global__ void modify_hue_kernel(png_bytep d_image, int width, int height, double *A) {
    // SEU CODIGO DO EP3 AQUI

    //calcula coordenadas do pixel para a thread CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //caso a thread esteja nos limites válidos da imagem
    if(idx < width && idy < height){
        //indice do pixel no vetor 'd_image', 3 canais RBG
        int pixel_idx = (idy * width + idx) * 3;
        //valores são lidos e normalizados no  intervalo [0, 1], / 255
        double r = d_image[pixel_idx] / 255.0;
        double g = d_image[pixel_idx + 1] / 255.0;
        double b = d_image[pixel_idx + 2] / 255.0;

        //novos valores de RGB

        double novo_r = r * A[0] + g * A[1] + b * A[2];
        double novo_g = r * A[3] + g * A[4] + b * A[5];
        double novo_b = r * A[6] + g * A[7] + b * A[8];

        //min e max garantem que os novos valores estejam dentro do intervalo [0, 1]
        novo_r = fmin(fmax(novo_r, 0.0), 1.0);
        novo_g = fmin(fmax(novo_g, 0.0), 1.0);
        novo_b = fmin(fmax(novo_b, 0.0), 1.0);

        //após garantir que os valores estão no intervalo, os valores são normalizados de volta para o intervalo [0, 255] e arredondando para o valor inteiro mais proximo e armazenados na imagem de saída (d_image)
        d_image[pixel_idx] = (png_byte)round(novo_r * 255.0);
        d_image[pixel_idx + 1] = (png_byte)round(novo_g * 255.0);
        d_image[pixel_idx + 2] = (png_byte)round(novo_b * 255.0);
    }
}



// Altera a matiz (hue) de uma imagem em paralelo
// Voce deve modificar essa funcao no EP3
// Função para calcular a matriz A com base no desvio de matiz (hue_diff)

//SEU CODIGO AQUI
void calculate_A(double *A, double hue_diff) {
    double c = cos(2 * M_PI * hue_diff);
    double s = sin(2 * M_PI * hue_diff);
    double one_third = 1.0 / 3.0;
    double sqrt_third = sqrt(one_third);

    // Preenche a matriz A com os valores calculados a partir de hue_diff
    A[0] = c + one_third * (1.0 - c);
    A[1] = one_third * (1.0 - c) - sqrt_third * s;
    A[2] = one_third * (1.0 - c) + sqrt_third * s;
    A[3] = A[2];
    A[4] = A[0];
    A[5] = A[1];
    A[6] = A[1];
    A[7] = A[2];
    A[8] = A[0];
}

// Função para alocar memória no dispositivo (GPU) e copiar os dados
void allocate_and_copy(double *h_A, png_bytep h_image, int width, int height, size_t image_size, double **d_A, png_bytep *d_image) {
    // Aloca memória para a matriz A no dispositivo (GPU)
    cudaMalloc((void **)d_A, sizeof(double) * 9);
    // Copia os dados da matriz A do host para o dispositivo
    cudaMemcpy(*d_A, h_A, sizeof(double) * 9, cudaMemcpyHostToDevice);

    // Aloca memória para a imagem de entrada no dispositivo (GPU)
    cudaMalloc((void **)d_image, image_size);
    // Copia os dados da imagem do host para o dispositivo
    cudaMemcpy(*d_image, h_image, image_size, cudaMemcpyHostToDevice);
}

// Função para copiar a imagem modificada de volta para o host (CPU) e liberar memória no dispositivo (GPU)
void copy_back_and_free(png_bytep h_image, png_bytep d_image, size_t image_size) {
    // Copia a imagem modificada do dispositivo para o host
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    // Libera a memória alocada no dispositivo
    cudaFree(d_image);
}


// Função para modificar o matiz em paralelo usando CUDA
void modify_hue(png_bytep h_image, int width, int height, size_t image_size, double hue_diff) {
    double A[9];
    calculate_A(A, hue_diff);

    double *d_A;
    png_bytep d_image;
    allocate_and_copy(A, h_image, width, height, image_size, &d_A, &d_image);

    // Configura as dimensões do grid e blocos para a chamada do kernel
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);

    // Chama o kernel para modificar o matiz em paralelo
    modify_hue_kernel<<<dim_grid, dim_block>>>(d_image, width, height, d_A);
    cudaDeviceSynchronize();

    // Copia a imagem modificada de volta para o host e libera a memória no dispositivo
    copy_back_and_free(h_image, d_image, image_size);
    cudaFree(d_A);
}


// Le imagem png de um arquivo de entrada para a memoria
void read_png_image(const char *filename,
                    png_bytep *image,
                    int *width,
                    int *height,
                    size_t *image_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Erro ao ler o arquivo de entrada %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Erro ao criar PNG read struct \n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Erro ao criar PNG info struct \n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Em caso de erro nas funcoes da libpng,
    // programa "pula" para este ponto de execucao
    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Erro ao ler imagem PNG \n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    // Verifica se imagem png possui o formato apropriado
    if ((color_type != PNG_COLOR_TYPE_RGB && color_type != PNG_COLOR_TYPE_GRAY)
        || bit_depth != 8) {
        printf("Formato PNG nao suportado, deve ser 8-bit RGB ou grayscale\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_read_update_info(png, info);

    // Alocacao de memoria para imagem e ponteiros para as linhas
    *image_size = png_get_rowbytes(png, info) * (*height);
    *image = (png_bytep) malloc(*image_size);

    png_bytep *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * (*height));
    for (int i = 0; i < *height; i++) {
        row_pointers[i] = *image + i * png_get_rowbytes(png, info);
    }

    // Leitura da imagem para a memoria
    png_read_image(png, row_pointers);

    // Finalizacao da leitura
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    free(row_pointers);
}

// Escreve imagem png da memoria para um arquivo de saida
void write_png_image(const char *filename,
                     png_bytep image,
                     int width,
                     int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Erro ao criar o arquivo de saida %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Erro ao criar PNG write struct \n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Erro ao criar PNG info struct.\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Em caso de erro nas funcoes da libpng,
    // programa "pula" para este ponto de execucao
    if (setjmp(png_jmpbuf(png))) {
        printf("Erro ao escrever imagem PNG \n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    // Configura o formato da imagem a ser criada
    png_set_IHDR(
        png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    // Criacao de ponteiros para as linhas
    png_bytep row_pointers[height];
    for (int i = 0; i < height; i++) {
        row_pointers[i] = &(image[i * width * 3]);
    }

    // Escrita da imagem a partir da memoria
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    // Finalizacao da escrita
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    png_bytep image;
    int width, height;
    size_t image_size;

    // Leitura e validacao dos parametros de entrada
    if (argc != 4) {
        printf("Uso: ./hue_modify <input_file> <output_file> <hue_diff>\n");
        printf("0.0 <= hue_diff <= 1.0\n");
        exit(EXIT_FAILURE);
    }

    double hue_diff;
    int ret = sscanf(argv[3], "%lf", &hue_diff);
    if (ret == 0 || ret == EOF) {
        fprintf(stderr, "Erro ao ler hue_diff\n");
        exit(EXIT_FAILURE);
    }

    if (hue_diff < 0.0 || hue_diff > 1.0) {
        fprintf(stderr, "hue_diff deve ser entre 0.0 e 1.0\n");
        exit(EXIT_FAILURE);
    }

    // Leitura da imagem para memoria
    read_png_image(argv[1], &image, &width, &height, &image_size);

    // Processamento da imagem (alteracao do hue)

    // Versao sequencial:
    modify_hue_seq(image, width, height, hue_diff);

    // // Versao paralela
    // modify_hue(image, width, height, image_size, hue_diff);

    // Escrita da imagem para arquivo
    write_png_image(argv[2], image, width, height);

    // Liberacao de memoria
    free(image);
    return 0;
}
