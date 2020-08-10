s/#undef  *\([ABCDEFGHIJKLMNOPQRSTUVWXYZ_]\)/#undef GMM_\1/
s/#undef  *\([abcdefghijklmnopqrstuvwxyz]\)/#undef _gmm_\1/
s/#define  *\([ABCDEFGHIJKLMNOPQRSTUVWXYZ_][abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_]*\)\(.*\)/#ifndef GMM_\1 \
#define GMM_\1 \2 \
#endif/
s/#define  *\([abcdefghijklmnopqrstuvwxyz][abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_]*\)\(.*\)/#ifndef _gmm_\1 \
#define _gmm_\1 \2 \
#endif/
