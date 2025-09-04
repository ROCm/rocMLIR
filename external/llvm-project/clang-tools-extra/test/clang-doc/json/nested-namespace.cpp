// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
<<<<<<< HEAD
// RUN: FileCheck %s < %t/nested/index.json --check-prefix=NESTED
// RUN: FileCheck %s < %t/nested/inner/index.json --check-prefix=INNER
=======
// RUN: FileCheck %s < %t/json/nested.json --check-prefix=NESTED
// RUN: FileCheck %s < %t/json/inner.json --check-prefix=INNER
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

namespace nested {
  int Global;
  namespace inner {
    int InnerGlobal;
  } // namespace inner
} // namespace nested

// NESTED:       "Variables": [
// NESTED-NEXT:    {
<<<<<<< HEAD
=======
// NESTED-NEXT:      "End": true,
// NESTED-NEXT:      "InfoType": "variable",
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
// NESTED-NEXT:      "IsStatic": false,
// NESTED-NEXT:      "Location": {
// NESTED-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// NESTED-NEXT:        "LineNumber": 7
// NESTED-NEXT:      },
// NESTED-NEXT:      "Name": "Global",
// NESTED-NEXT:      "Namespace": [
// NESTED-NEXT:        "nested"
// NESTED-NEXT:      ],

// INNER:       "Variables": [
// INNER-NEXT:    {
<<<<<<< HEAD
=======
// INNER-NEXT:      "End": true,
// INNER-NEXT:      "InfoType": "variable",
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
// INNER-NEXT:      "IsStatic": false,
// INNER-NEXT:      "Location": {
// INNER-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// INNER-NEXT:        "LineNumber": 9
// INNER-NEXT:      },
// INNER-NEXT:      "Name": "InnerGlobal",
// INNER-NEXT:      "Namespace": [
// INNER-NEXT:        "inner",
// INNER-NEXT:        "nested"
// INNER-NEXT:      ],
