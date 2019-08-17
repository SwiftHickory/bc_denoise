#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "bc_denoise.h"

#define LINESIZE 512

extern Param_t* Param;

int bc_abort(const char* fname, const char* perror_msg, const char* format, ...) {
    int my_errno = errno;

    fprintf(stderr, "\nbc_denoise called solver_abort()");

    if (fname != NULL) {
        fprintf(stderr, "from %s: \n", fname);
    }

    if (format != NULL) {
        va_list ap;

        va_start(ap, format);
        vfprintf(stderr, format, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    if (perror_msg != NULL) {
        fprintf(stderr, "%s: %s\n", perror_msg, strerror(my_errno));
    }

    fprintf(stderr, "\n");

    exit(1);

    return -1;
}

/**
 * Parse a text file and return the value of a match string.
 */
int parsetext(FILE* fp, const char* querystring, const char type, void* result) {
    const static char delimiters[] = " =\n\t";

    int32_t res = 0, found = 0;

    // start from the beginning
    rewind(fp);

    //look for the string until found
    while (!found) {
        char line[LINESIZE];
        char *name, *value;

        /* Read in one line */
        if (fgets(line, LINESIZE, fp) == NULL)
            break;

        name = strtok(line, delimiters);
        if ((name != NULL) && (strcmp(name, querystring) == 0)) {
            found = 1;
            value = strtok(NULL, delimiters);

            switch (type) {
                case 'i':
                    res = sscanf(value, "%d", (int *)result);
                    break;

                case 'f':
                    res = sscanf(value, "%f", (float *)result);
                    break;

                case 'd':
                    res = sscanf(value, "%lf", (double *)result);
                    break;

                case 's':
                    res = 1;
                    strcpy((char *)result, value);
                    break;

                case 'u':
                    res = sscanf(value, "%u", (uint32_t *)result);
                    break;

                default:
                    fprintf(stderr, "parsetext: unknown type %c\n", type);
                    return -1;
            }
        }
    }

    return (res == 1) ? 0 : -1;
}