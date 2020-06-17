## vectorize, load gcc/8.3.0, USE jobsimd.sh
#CC = gcc
#CFLAGS = -fopt-info-vec-optimized -march=native -O3
#LDFLAGS =
#LIBS =

#APP = diffusion

## OpenMP, load gcc/8.3.0, USE jobomp.sh
#CC = gcc
#CFLAGS = -O3 -fopenmp
#LDFLAGS = -fopenmp
#LIBS =

#APP = diffusion

## OpenMP + vectorize, load gcc/8.3.0, USE jobomp.sh
#CC = gcc
#CFLAGS = -fopt-info-vec-optimized -march=native -O3 -fopenmp
#LDFLAGS = -fopenmp
#LIBS =

#APP = diffusion

## OpenACC, load pgi, USE joboac.sh
CC = pgcc
CFLAGS = -O3 -acc -Minfo=accel
LDFLAGS = -acc
LIBS =

APP = diffusion

## Cuda, load cuda, USE jobcuda.sh
#CC = nvcc
##CFLAGS = -O2 -g -arch sm_60
#CFLAGS = -O3 -g
##LDFLAGS = -arch sm_60
#LDFLAGS =
#LIBS =

#APP = diffusion

## MPI
#CC = mpicc
#CFLAGS = -O3 -g
#LDFLAGS =
#LIBS =

#APP = diffusion

## common part
OBJS = $(APP).o

all: $(APP)

$(APP): $(OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f *~
	rm -f $(APP)
