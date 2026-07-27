switch("mm", "orc")
switch("opt", "speed")
switch("threads", "on")
switch("d", "release")
# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config
