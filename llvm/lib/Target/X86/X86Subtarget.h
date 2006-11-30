//=====---- X86Subtarget.h - Define Subtarget for the X86 -----*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef X86SUBTARGET_H
#define X86SUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

#include <string>

namespace llvm {
class Module;
class GlobalValue;

class X86Subtarget : public TargetSubtarget {
public:
  enum AsmWriterFlavorTy {
    att, intel, unset
  };

protected:
  enum X86SSEEnum {
    NoMMXSSE, MMX, SSE1, SSE2, SSE3
  };

  enum X863DNowEnum {
    NoThreeDNow, ThreeDNow, ThreeDNowA
  };

  /// AsmFlavor - Which x86 asm dialect to use.
  AsmWriterFlavorTy AsmFlavor;

  /// X86SSELevel - MMX, SSE1, SSE2, SSE3, or none supported.
  X86SSEEnum X86SSELevel;

  /// X863DNowLevel - 3DNow or 3DNow Athlon, or none supported.
  X863DNowEnum X863DNowLevel;

  /// HasX86_64 - True if the processor supports X86-64 instructions.
  bool HasX86_64;

  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// Min. memset / memcpy size that is turned into rep/movs, rep/stos ops.
  unsigned MinRepStrSizeThreshold;

private:
  /// Is64Bit - True if the processor supports 64-bit instructions and module
  /// pointer size is 64 bit.
  bool Is64Bit;

public:
  enum {
    isELF, isCygwin, isDarwin, isWindows
  } TargetType;

  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  X86Subtarget(const Module &M, const std::string &FS, bool is64Bit);

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// getMinRepStrSizeThreshold - Returns the minimum memset / memcpy size
  /// required to turn the operation into a X86 rep/movs or rep/stos
  /// instruction. This is only used if the src / dst alignment is not DWORD
  /// aligned.
  unsigned getMinRepStrSizeThreshold() const { return MinRepStrSizeThreshold; }

  /// ParseSubtargetFeatures - Parses features string setting specified
  /// subtarget options.  Definition of function is auto generated by tblgen.
  void ParseSubtargetFeatures(const std::string &FS, const std::string &CPU);

  /// AutoDetectSubtargetFeatures - Auto-detect CPU features using CPUID
  /// instruction.
  void AutoDetectSubtargetFeatures();

  bool is64Bit() const { return Is64Bit; }

  bool hasMMX() const { return X86SSELevel >= MMX; }
  bool hasSSE1() const { return X86SSELevel >= SSE1; }
  bool hasSSE2() const { return X86SSELevel >= SSE2; }
  bool hasSSE3() const { return X86SSELevel >= SSE3; }
  bool has3DNow() const { return X863DNowLevel >= ThreeDNow; }
  bool has3DNowA() const { return X863DNowLevel >= ThreeDNowA; }

  bool isFlavorAtt() const { return AsmFlavor == att; }
  bool isFlavorIntel() const { return AsmFlavor == intel; }

  bool isTargetDarwin() const { return TargetType == isDarwin; }
  bool isTargetELF() const { return TargetType == isELF; }
  bool isTargetWindows() const { return TargetType == isWindows; }
  bool isTargetCygwin() const { return TargetType == isCygwin; }

  /// True if accessing the GV requires an extra load. For Windows, dllimported
  /// symbols are indirect, loading the value at address GV rather then the
  /// value of GV itself. This means that the GlobalAddress must be in the base
  /// or index register of the address, not the GV offset field.
  bool GVRequiresExtraLoad(const GlobalValue* GV, bool isDirectCall) const;
};

namespace X86 {
  /// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in
  /// the specified arguments.  If we can't run cpuid on the host, return true.
  bool GetCpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                       unsigned *rECX, unsigned *rEDX);
}

} // End llvm namespace

#endif
