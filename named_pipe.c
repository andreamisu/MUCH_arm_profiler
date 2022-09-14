// C program to implement one side of FIFO
// This side writes first, then reads
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int pipe_comunication(int pipeno)
{
    int fd;

    // FIFO file path
    char * myfifo = "/mypipe";

    // Creating the named file(FIFO)
    // mkfifo(<pathname>, <permission>)
    //mkfifo(myfifo, 0660);

    char arr1[80], arr2[80];
    while (1)
    {
        // Open FIFO for write only
        //fd = open(pipeno, O_WRONLY);

        // Take an input arr2ing from user.
        // 80 is maximum length
        fgets(arr2, 80, stdin);

        // Write the input arr2ing on FIFO
        // and close it
        write(pipeno, arr2, strlen(arr2)+1);
        close(fd);

    }
    return 0;
}