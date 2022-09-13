if __name__ == '__main__':
    try:
        r_filepath = '/home/m34ferna/src/flink-setup/bash-scripts/flink_kakfa_metrics_list.txt'
        w_filepath = '/home/m34ferna/src/flink-setup/bash-scripts/export_sql_queries.txt'
        read_fp = open(r_filepath, 'r')
        write_fp = open(w_filepath, 'w')
        line = read_fp.readline()
        cnt = 0
        while line:
            metric_name = line.strip(' \t\n\r')
            write_fp.write(
                "$INFLUX_CMD -database 'flink' -execute \"SELECT * FROM \\\"" + metric_name +
                "\\\" WHERE time > now() - $1m\" \\"
                "\n-format csv > " + metric_name + "_$2_`date -u +%Y_%m_%d`.csv\n\n")
            cnt += 1
            line = read_fp.readline()

        print("Read " + str(cnt) + " lines.")
    finally:
        read_fp.close()
        write_fp.close()
