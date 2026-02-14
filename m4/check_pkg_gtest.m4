# CHECK_PKG_GTEST([action-if-found], [action-if-not-found])
# --------------------------------------------------------
AC_DEFUN([CHECK_PKG_GTEST], [
    AC_ARG_WITH([gtest],
        [AS_HELP_STRING([--with-gtest=PATH],
            [Path to GoogleTest installation])],
        [gtest_path="$withval"],
        [gtest_path=""])

    AC_ARG_ENABLE([gtest],
        [AS_HELP_STRING([--enable-gtest],
            [Enable GoogleTest/GoogleMock unit tests (default: no)])],
        [enable_gtest="$enableval"],
        [enable_gtest="no"])

    have_gtest=no
    GTEST_CPPFLAGS=""
    GTEST_LDFLAGS=""
    GTEST_LIBS=""

    AS_IF([test "x$enable_gtest" != "xno"], [
        save_CPPFLAGS="$CPPFLAGS"
        save_LDFLAGS="$LDFLAGS"
        save_LIBS="$LIBS"

        AS_IF([test "x$gtest_path" != "x"], [
            GTEST_CPPFLAGS="-I$gtest_path/include"
            GTEST_LDFLAGS="-L$gtest_path/lib"
        ])

        CPPFLAGS="$CPPFLAGS $GTEST_CPPFLAGS"
        LDFLAGS="$LDFLAGS $GTEST_LDFLAGS"

        AC_CHECK_HEADERS([gtest/gtest.h gmock/gmock.h], [
            AC_CHECK_LIB([gtest], [main], [
                AC_CHECK_LIB([gmock], [main], [
                    have_gtest=yes
                    GTEST_LIBS="-lgtest -lgmock -lpthread"
                ], [], [-lgtest -lpthread])
            ], [], [-lpthread])
        ])

        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"
        LIBS="$save_LIBS"

        AS_IF([test "x$have_gtest" = "xyes"], [
            AC_MSG_NOTICE([GoogleTest/GoogleMock found])
            $1
        ], [
            AS_IF([test "x$enable_gtest" = "xyes"], [
                AC_MSG_ERROR([GoogleTest/GoogleMock requested but not found])
            ], [
                AC_MSG_NOTICE([GoogleTest/GoogleMock not found, disabling GoogleTest tests])
            ])
            $2
        ])
    ])

    AC_SUBST([GTEST_CPPFLAGS])
    AC_SUBST([GTEST_LDFLAGS])
    AC_SUBST([GTEST_LIBS])
    AM_CONDITIONAL([ENABLE_GTEST], [test "x$have_gtest" = "xyes"])
])
