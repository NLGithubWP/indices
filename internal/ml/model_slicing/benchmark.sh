#!/bin/bash

# Number of times to run the SQL command
count=300

# Database connection parameters
host=localhost
port=28814
user=postgres
dbname=pg_extension

# SQL command to be executed
sql_command="SELECT run_inference_profiling(4, 'avazu', '{}', '/project/Trails/internal/ml/model_selection/config.ini', '/project/Trails/internal/ml/model_slicing/data/avazu_padding.json', '/project/tensor_log/avazu/dnn_K16', '', 10000);"

# Create a temporary file to hold the SQL commands
tmpfile=$(mktemp /tmp/sql_commands.XXXXXX)

# Add the SQL command to the temporary file $count times
for ((i=1; i<=count; i++))
do
    echo "$sql_command" >> "$tmpfile"
done

# Execute the SQL commands in the temporary file
PGPASSWORD=your_password psql -h $host -p $port -U $user -d $dbname -f "$tmpfile"

# Remove the temporary file
rm "$tmpfile"

echo "Done."