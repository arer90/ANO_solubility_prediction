# 1. Verify Strawberry Perl installation
perl --version

# 2. Check Circos module dependencies
perl circos -modules

# 3. Auto-install missing modules
perl -e "$output = `perl circos -modules`; while($output =~ /missing\s+(\S+)/g) { system('cpan', $1); }"

# [Alternative: Manual module installation]
# perl -MCPAN -e shell
# install Module::Name
# exit

# 4. Diagnose GD library
perl gddiag

# 5. Run Circos
circos -conf circos.cnf

# [If paranoid error occurs]
# Use this only when you get a paranoid-specific error
circos -noparanoid -conf circos.cnf